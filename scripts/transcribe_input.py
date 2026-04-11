import os
import sys
from faster_whisper import WhisperModel
from rich.console import Console

console = Console()


def transcribe(file_path):
    if not os.path.exists(file_path):
        console.print(f"[red]Error: File {file_path} not found.[/red]")
        return

    console.print(
        f"[blue]Transcribing {file_path} using faster-whisper (base)...[/blue]"
    )

    # Run on CPU for compatibility, using int8 quantization to be fast
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(file_path, beam_size=5)

    console.print(
        f"[green]Detected language '{info.language}' with probability {info.language_probability:.2f}[/green]"
    )

    full_text = ""
    for segment in segments:
        console.print(f"[[{segment.start:.2f}s -> {segment.end:.2f}s]] {segment.text}")
        full_text += segment.text + " "

    return full_text.strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe_input.py <file_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    transcript = transcribe(input_file)

    if transcript:
        output_file = "latest_transcript.txt"
        with open(output_file, "w") as f:
            f.write(transcript)
        console.print(
            f"\n[bold green]Success![/bold green] Transcript saved to {output_file}"
        )
        print("\n--- TRANSCRIPT START ---")
        print(transcript)
        print("--- TRANSCRIPT END ---")
