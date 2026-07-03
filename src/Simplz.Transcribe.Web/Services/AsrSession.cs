using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Microsoft.Extensions.Options;

namespace Simplz.Transcribe.Web.Services;

public enum AsrEventType { Partial, Final, Error }

/// <param name="Text">For <see cref="AsrEventType.Partial"/> this is an incremental
/// delta to append; for <see cref="AsrEventType.Final"/> the full transcript.</param>
public sealed record AsrEvent(AsrEventType Type, string Text);

/// <summary>Creates transcription sessions against the ASR backend.</summary>
public sealed class AsrClient(IOptions<AsrOptions> options)
{
    public async Task<AsrSession> StartSessionAsync(CancellationToken ct)
    {
        var opts = options.Value;
        var ws = new ClientWebSocket();
        try
        {
            await ws.ConnectAsync(new Uri(opts.WebSocketUrl), ct);
        }
        catch
        {
            ws.Dispose();
            throw;
        }
        var session = new AsrSession(ws, opts.Model);
        await session.InitializeAsync(ct);
        return session;
    }
}

/// <summary>
/// One live transcription over the vLLM Realtime WebSocket protocol:
/// send PCM16LE mono 16 kHz audio, read partial/final events from <see cref="Events"/>.
/// </summary>
public sealed class AsrSession : IAsyncDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new(JsonSerializerDefaults.Web);

    private readonly ClientWebSocket _ws;
    private readonly string _model;
    private readonly Channel<AsrEvent> _events = Channel.CreateUnbounded<AsrEvent>();
    private Task? _receiveLoop;

    internal AsrSession(ClientWebSocket ws, string model)
    {
        _ws = ws;
        _model = model;
    }

    public ChannelReader<AsrEvent> Events => _events.Reader;

    internal async Task InitializeAsync(CancellationToken ct)
    {
        await SendJsonAsync(new { type = "session.update", model = _model }, ct);
        _receiveLoop = Task.Run(() => ReceiveLoopAsync(ct), ct);
    }

    public ValueTask SendAudioAsync(ReadOnlyMemory<byte> pcm16, CancellationToken ct)
        => new(SendJsonAsync(new
        {
            type = "input_audio_buffer.append",
            audio = Convert.ToBase64String(pcm16.Span),
        }, ct));

    public Task CommitAsync(CancellationToken ct)
        => SendJsonAsync(new { type = "input_audio_buffer.commit", final = true }, ct);

    private async Task SendJsonAsync(object message, CancellationToken ct)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(message, JsonOpts);
        await _ws.SendAsync(bytes, WebSocketMessageType.Text, endOfMessage: true, ct);
    }

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        var buffer = new byte[64 * 1024];
        var message = new MemoryStream();
        try
        {
            while (_ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                message.SetLength(0);
                WebSocketReceiveResult result;
                do
                {
                    result = await _ws.ReceiveAsync(buffer, ct);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        _events.Writer.TryComplete();
                        return;
                    }
                    message.Write(buffer, 0, result.Count);
                } while (!result.EndOfMessage);

                HandleMessage(Encoding.UTF8.GetString(message.GetBuffer(), 0, (int)message.Length));
            }
            _events.Writer.TryComplete();
        }
        catch (Exception ex) when (ex is OperationCanceledException or WebSocketException)
        {
            _events.Writer.TryWrite(new AsrEvent(AsrEventType.Error, "ASR backend connection lost"));
            _events.Writer.TryComplete();
        }
    }

    private void HandleMessage(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        switch (root.GetProperty("type").GetString())
        {
            case "transcription.delta":
                _events.Writer.TryWrite(new AsrEvent(AsrEventType.Partial,
                    root.GetProperty("delta").GetString() ?? ""));
                break;
            case "transcription.done":
                _events.Writer.TryWrite(new AsrEvent(AsrEventType.Final,
                    root.GetProperty("text").GetString() ?? ""));
                _events.Writer.TryComplete();
                break;
            case "error":
                _events.Writer.TryWrite(new AsrEvent(AsrEventType.Error,
                    root.TryGetProperty("error", out var err) ? err.GetString() ?? "" : "unknown error"));
                _events.Writer.TryComplete();
                break;
            case "session.created":
                break;
        }
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_ws.State == WebSocketState.Open)
            {
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
                await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", cts.Token);
            }
        }
        catch (Exception ex) when (ex is OperationCanceledException or WebSocketException)
        {
            // best-effort close
        }
        _ws.Dispose();
    }
}
