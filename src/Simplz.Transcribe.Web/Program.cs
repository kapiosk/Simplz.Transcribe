using System.Diagnostics;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Simplz.Transcribe.Web.Services;

var builder = WebApplication.CreateBuilder(args);

builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = null); // large media uploads
builder.Services.Configure<AsrOptions>(builder.Configuration.GetSection("Asr"));
builder.Services.AddSingleton<AsrClient>();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();
app.UseWebSockets();

app.MapGet("/api/health", () => Results.Ok(new { status = "ok" }));

// ---------------------------------------------------------------------------
// Live microphone: browser WebSocket.
//   C->S text  {"type":"start"} ... binary PCM16LE mono 16 kHz frames ... {"type":"stop"}
//   S->C text  {"type":"partial","text":<delta>} | {"type":"final","text":<full>} | {"type":"error","text":...}
// ---------------------------------------------------------------------------
app.Map("/ws/transcribe", async (HttpContext context, AsrClient asrClient) =>
{
    if (!context.WebSockets.IsWebSocketRequest)
    {
        return Results.BadRequest("WebSocket request expected");
    }

    using var ws = await context.WebSockets.AcceptWebSocketAsync();
    var ct = context.RequestAborted;

    AsrSession session;
    try
    {
        session = await asrClient.StartSessionAsync(ct);
    }
    catch (Exception ex) when (ex is not OperationCanceledException)
    {
        await SendBrowserEventAsync(ws, "error", $"ASR backend unavailable: {ex.Message}", ct);
        await ws.CloseAsync(WebSocketCloseStatus.InternalServerError, "asr unavailable", CancellationToken.None);
        return Results.Empty;
    }

    await using (session)
    {
        var forward = Task.Run(async () =>
        {
            await foreach (var evt in session.Events.ReadAllAsync(ct))
            {
                var type = evt.Type switch
                {
                    AsrEventType.Partial => "partial",
                    AsrEventType.Final => "final",
                    _ => "error",
                };
                await SendBrowserEventAsync(ws, type, evt.Text, ct);
            }
        }, ct);

        var buffer = new byte[64 * 1024];
        try
        {
            var committed = false;
            while (ws.State == WebSocketState.Open)
            {
                var result = await ws.ReceiveAsync(buffer, ct);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    break;
                }
                if (result.MessageType == WebSocketMessageType.Binary)
                {
                    await session.SendAudioAsync(buffer.AsMemory(0, result.Count), ct);
                }
                else
                {
                    var text = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    if (text.Contains("\"stop\"", StringComparison.Ordinal))
                    {
                        await session.CommitAsync(ct);
                        committed = true;
                    }
                    // "start" needs no action: the session is already open.
                }
            }

            if (!committed)
            {
                await session.CommitAsync(ct); // browser closed without "stop"
            }
            await forward; // drain remaining partials + final
            if (ws.State == WebSocketState.Open)
            {
                await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None);
            }
        }
        catch (Exception ex) when (ex is OperationCanceledException or WebSocketException)
        {
            // browser went away — session disposal tears down the backend connection
        }
    }
    return Results.Empty;
});

// ---------------------------------------------------------------------------
// File/video upload: raw request body -> ffmpeg -> ASR, response is SSE.
//   curl -N --data-binary @sample.mp3 http://localhost:8080/api/transcribe-file
// ---------------------------------------------------------------------------
app.MapPost("/api/transcribe-file", async (HttpContext context, AsrClient asrClient) =>
{
    var ct = context.RequestAborted;
    var response = context.Response;
    response.Headers.ContentType = "text/event-stream";
    response.Headers.CacheControl = "no-cache";

    AsrSession session;
    try
    {
        session = await asrClient.StartSessionAsync(ct);
    }
    catch (Exception ex) when (ex is not OperationCanceledException)
    {
        await WriteSseAsync(response, "error", $"ASR backend unavailable: {ex.Message}", ct);
        return;
    }

    await using (session)
    {
        using var ffmpeg = FfmpegDecoder.Start();
        var stderrTask = ffmpeg.StandardError.ReadToEndAsync(ct);

        // Upload -> ffmpeg stdin. Failures just close stdin so the pipeline drains.
        var feedInput = Task.Run(async () =>
        {
            try
            {
                await context.Request.Body.CopyToAsync(ffmpeg.StandardInput.BaseStream, ct);
            }
            catch (IOException)
            {
                // ffmpeg exited early (e.g. unsupported input) — surfaced via stderr below
            }
            finally
            {
                ffmpeg.StandardInput.Close();
            }
        }, ct);

        // ffmpeg stdout (PCM) -> ASR, paced to at most ~16x real-time so the
        // sidecar's input buffer stays bounded on long inputs.
        const int maxBytesPerSecond = 16_000 * 2 * 16;
        var pump = Task.Run(async () =>
        {
            var pcm = new byte[32 * 1024];
            long sent = 0;
            var clock = Stopwatch.StartNew();
            int n;
            while ((n = await ffmpeg.StandardOutput.BaseStream.ReadAsync(pcm, ct)) > 0)
            {
                await session.SendAudioAsync(pcm.AsMemory(0, n), ct);
                sent += n;
                var earliest = TimeSpan.FromSeconds((double)sent / maxBytesPerSecond);
                if (clock.Elapsed < earliest)
                {
                    await Task.Delay(earliest - clock.Elapsed, ct);
                }
            }
            await feedInput;
            await session.CommitAsync(ct);
            return sent;
        }, ct);

        try
        {
            await foreach (var evt in session.Events.ReadAllAsync(ct))
            {
                var type = evt.Type switch
                {
                    AsrEventType.Partial => "partial",
                    AsrEventType.Final => "final",
                    _ => "error",
                };
                await WriteSseAsync(response, type, evt.Text, ct);
            }

            // Events ended. Normally the pump has finished (commit precedes the
            // final event); if the backend errored early, don't wait on it.
            if (pump.IsCompleted)
            {
                var pcmBytes = await pump;
                if (pcmBytes == 0)
                {
                    var stderr = await stderrTask;
                    await WriteSseAsync(response, "error",
                        $"could not decode any audio from the uploaded file: {Truncate(stderr, 500)}", ct);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // client cancelled the request
        }
        finally
        {
            if (!ffmpeg.HasExited)
            {
                ffmpeg.Kill(entireProcessTree: true); // also unblocks a still-running pump
            }
            await Task.WhenAny(pump); // observe pump faults without throwing
        }
    }
});

app.Run();

static async Task SendBrowserEventAsync(WebSocket ws, string type, string text, CancellationToken ct)
{
    if (ws.State != WebSocketState.Open)
    {
        return;
    }
    var payload = JsonSerializer.SerializeToUtf8Bytes(new { type, text });
    await ws.SendAsync(payload, WebSocketMessageType.Text, endOfMessage: true, ct);
}

static async Task WriteSseAsync(HttpResponse response, string type, string text, CancellationToken ct)
{
    var json = JsonSerializer.Serialize(new { type, text });
    await response.WriteAsync($"data: {json}\n\n", ct);
    await response.Body.FlushAsync(ct);
}

static string Truncate(string value, int max)
    => value.Length <= max ? value : value[..max] + "…";
