#!/usr/bin/env python3
"""
Main Entry Point for Battlefield Reconnaissance Video Agent System
Provides CLI interface for launching UI or running analysis directly
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def launch_ui():
    """Launch Gradio Web UI"""
    from ui.gradio_app import create_ui

    print("\n" + "="*60)
    print("Battlefield Reconnaissance Video Agent - Web UI")
    print("="*60 + "\n")

    ui = create_ui()
    ui.launch()


def analyze_video(video_path: str, segment_duration: int = 30, output_dir: Optional[str] = None, preload: bool = True):
    """Analyze video via CLI"""
    from core_src.model_manager import get_global_model_manager

    print("\n" + "="*60)
    print("Battlefield Reconnaissance Video Agent - CLI Analysis")
    print("="*60 + "\n")

    # Get model manager and pre-load models
    model_manager = get_global_model_manager()

    if preload:
        print("Pre-loading models...")
        model_manager.load_all_models(load_agent=False)  # Don't load agent for video analysis

    # Initialize system with pre-loaded models
    system = model_manager.get_video_system()

    # Analyze video
    results = system.analyze_video(
        video_path=video_path,
        segment_duration=segment_duration
    )

    # Export results if output dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export metadata
        metadata_file = output_path / f"{results['video_id']}_metadata.json"
        system.export_results(results['video_id'], str(metadata_file))

        print(f"\n✅ Results exported to: {metadata_file}")

    # Print summary
    summary = results["summary"]
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)
    print(f"Video ID: {results['video_id']}")
    print(f"Duration: {summary['video_length']:.1f}s")
    print(f"Segments: {summary['num_segments']}")
    print(f"\nObject Counts:")
    # Dynamically print object counts for all detected classes
    for obj_class, count in summary['object_counts'].items():
        print(f"  - {obj_class.capitalize()}s: {count}")
    print(f"\nStream URL: {summary['stream_url']}")
    print("="*60 + "\n")

    # Cleanup
    system.cleanup()


def run_agent_query(query: str, video_id: Optional[str] = None, preload: bool = True):
    """Run agent query via CLI"""
    from core_src.model_manager import get_global_model_manager

    print("\n" + "="*60)
    print("Battlefield Reconnaissance Video Agent - Query")
    print("="*60 + "\n")

    # Get model manager and pre-load models
    model_manager = get_global_model_manager()

    if preload:
        print("Pre-loading models...")
        model_manager.load_all_models(load_agent=True)  # Load agent for queries

    # Create agent with pre-loaded model
    agent = model_manager.get_agent()

    # Set video context if provided
    if video_id:
        agent.set_video_context(video_id)
        print(f"Video context set to: {video_id}\n")

    # Run query
    print(f"Query: {query}\n")
    result = agent.run(query)

    print("\n" + "="*60)
    print("Agent Response:")
    print("="*60)
    print(result)
    print("="*60 + "\n")


def add_pdf(pdf_path: str):
    """Add PDF to RAG system via CLI"""
    from tools.pdf_rag_tool import get_rag_system

    print("\n" + "="*60)
    print("Adding PDF to RAG System")
    print("="*60 + "\n")

    rag_system = get_rag_system()
    num_chunks = rag_system.add_pdf(pdf_path)

    print(f"\n✅ Successfully added {num_chunks} chunks from {pdf_path}")
    print("="*60 + "\n")


def check_environment():
    """Check if required environment variables and paths are set"""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60 + "\n")

    # Check local storage
    local_db_path = Path("./data/local_videodb")
    if local_db_path.exists():
        print(f"✅ Local video database exists at: {local_db_path}")
        db_file = local_db_path / "videodb.sqlite"
        if db_file.exists():
            print(f"   Database file size: {db_file.stat().st_size / 1024:.1f} KB")
    else:
        print(f"ℹ️  Local video database will be created at: {local_db_path}")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA is not available, will use CPU")
    except ImportError:
        print("⚠️  PyTorch not installed")

    # Check paths
    current_dir = Path(__file__).parent
    print(f"\n📁 Project Directory: {current_dir}")

    required_dirs = ["config", "core", "tools", "agent", "ui", "utils"]
    for dir_name in required_dirs:
        if (current_dir / dir_name).exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ (missing)")

    print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Battlefield Reconnaissance Video Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Web UI
  python main.py ui

  # Analyze video
  python main.py analyze --video path/to/video.mp4 --output results/

  # Query agent
  python main.py query --query "How many tanks?" --video-id m-xxxxx

  # Add PDF to RAG
  python main.py add-pdf --pdf path/to/document.pdf

  # Check environment
  python main.py check
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # UI command
    subparsers.add_parser("ui", help="Launch Gradio Web UI")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze video")
    analyze_parser.add_argument("--video", required=True, help="Path to video file")
    analyze_parser.add_argument("--segment-duration", type=int, default=30,
                               help="Segment duration in seconds (default: 30)")
    analyze_parser.add_argument("--output", help="Output directory for results")
    analyze_parser.add_argument("--no-preload", action="store_true",
                               help="Skip model pre-loading (not recommended)")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run agent query")
    query_parser.add_argument("--query", required=True, help="Query text")
    query_parser.add_argument("--video-id", help="Video ID for context")
    query_parser.add_argument("--no-preload", action="store_true",
                             help="Skip model pre-loading (not recommended)")

    # Add PDF command
    pdf_parser = subparsers.add_parser("add-pdf", help="Add PDF to RAG system")
    pdf_parser.add_argument("--pdf", required=True, help="Path to PDF file")

    # Check command
    subparsers.add_parser("check", help="Check environment and setup")

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "ui":
        launch_ui()

    elif args.command == "analyze":
        analyze_video(
            video_path=args.video,
            segment_duration=args.segment_duration,
            output_dir=args.output,
            preload=not args.no_preload
        )

    elif args.command == "query":
        run_agent_query(
            query=args.query,
            video_id=args.video_id,
            preload=not args.no_preload
        )

    elif args.command == "add-pdf":
        add_pdf(pdf_path=args.pdf)

    elif args.command == "check":
        check_environment()

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
