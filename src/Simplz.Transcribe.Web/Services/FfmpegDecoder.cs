using System.Diagnostics;

namespace Simplz.Transcribe.Web.Services;

/// <summary>
/// Decodes any audio/video input to raw PCM16LE mono 16 kHz via an ffmpeg pipe.
/// </summary>
public static class FfmpegDecoder
{
    public static Process Start()
    {
        var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = "-hide_banner -loglevel error -i pipe:0 -vn -f s16le -acodec pcm_s16le -ac 1 -ar 16000 pipe:1",
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            },
            EnableRaisingEvents = true,
        };
        process.Start();
        return process;
    }
}
