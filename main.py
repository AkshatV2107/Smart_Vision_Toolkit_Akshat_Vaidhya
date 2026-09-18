from pathlib import Path
from vision_toolkit.pipeline import run_pipeline
from vision_toolkit.synthetic import create_sample_image

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "sample_data" / "sample_scene.png"
OUTPUT = ROOT / "outputs"

def main():
    INPUT.parent.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)

    create_sample_image(INPUT)
    result = run_pipeline(INPUT, OUTPUT)

    print("SMART VISION TOOLKIT")
    print("-" * 60)
    for key, value in result["stats"].items():
        print(f"{key}: {value}")
    print(f"Objects detected: {result['object_count']}")
    print(f"Outputs written to: {OUTPUT}")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
