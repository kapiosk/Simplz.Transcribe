namespace Simplz.Transcribe.Web.Services;

public sealed class AsrOptions
{
    /// <summary>vLLM-Realtime-compatible WebSocket endpoint (sidecar or a real vLLM box).</summary>
    public string WebSocketUrl { get; set; } = "ws://asr:8000/v1/realtime";

    public string Model { get; set; } = "mistralai/Voxtral-Mini-4B-Realtime-2602";
}
