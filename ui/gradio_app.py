"""
Gradio Web UI for Battlefield Reconnaissance Video Agent
Provides interface for video upload, analysis, and agent interaction
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json
import yaml
import tempfile

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Add map_mcp to path
map_mcp_path = parent_dir / "map_mcp" / "src"
sys.path.insert(0, str(map_mcp_path))

import gradio as gr
# ChatMessage removed - using tuple format for Gradio compatibility

# Import map_mcp for war game functionality
from map_mcp.wargame_map import WarGameMap
from map_mcp.wargame_ui import WarGameUI, _get_temp_dir, _setup_static_paths
from map_mcp.military_symbols import UnitType, Echelon, Affiliation

# Import core system
from core_src.video_analysis_system import VideoAnalysisSystem
from agent.battlefield_agent import BattlefieldReconnaissanceAgent
from core_src.model_manager import get_global_model_manager
from core_src.context_scanner import ContextScanner
from core_src.collection_manager import CollectionManager, get_collection_state

# Import tool setters for context management
from tools.videodb_query_tool import set_selected_video_ids, set_active_collections
from tools.pdf_rag_tool import set_selected_pdf_files, get_rag_system


class BattlefieldVideoAgentUI:
    """
    Gradio UI for Battlefield Reconnaissance Video Agent System
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize UI

        Args:
            config_path: Path to configuration directory
        """
        if config_path is None:
            config_path = parent_dir / "config"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self.config = self._load_config()

        # Get global model manager
        self.model_manager = get_global_model_manager(config_path)

        # Initialize systems (using pre-loaded models from manager)
        self.video_system = None
        self.agent = None
        self.current_video_id = None

        # Initialize context scanner
        self.context_scanner = ContextScanner()

        # Initialize collection manager
        self.collection_manager = CollectionManager()

        # Ensure default collection exists
        self.collection_manager.ensure_default_collection_exists()

        # Get global collection state
        self.collection_state = get_collection_state()

        # Initialize tools with default active collections
        set_active_collections(self.collection_state.get_active_collections())

        # Selected contexts (for multi-context support)
        self.selected_video_ids = []
        self.selected_pdf_files = []

        # Agent execution control
        self.stop_requested = False
        self.agent_thread = None

        # Initialize war game map (tiles directory can be configured)
        tiles_dir = self.config_path.parent / "map_mcp" / "tiles"
        if not tiles_dir.exists():
            tiles_dir = None  # Use placeholders if no tiles
        self.wargame_ui = WarGameUI(tiles_dir=tiles_dir)

    def _load_config(self) -> Dict[str, Any]:
        """Load UI configuration"""
        try:
            with open(self.config_path / "agent_config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _init_video_system(self):
        """Initialize video system using pre-loaded models"""
        if self.video_system is None:
            if not self.model_manager.check_models_ready():
                raise RuntimeError("Models not loaded. Please call load_all_models() first.")
            print("Initializing Video Analysis System...")
            self.video_system = self.model_manager.get_video_system()

    def _init_agent(self):
        """Initialize agent using pre-loaded model"""
        if self.agent is None:
            if not self.model_manager.check_agent_ready():
                raise RuntimeError("Agent model not loaded. Please call load_all_models() first.")
            print("Initializing Battlefield Agent...")
            self.agent = self.model_manager.get_agent()

    def _get_class_emoji(self, obj_class: str) -> str:
        """
        Get emoji for an object class (for display purposes).

        Args:
            obj_class: Object class name (e.g., "truck", "tank", "soldier")

        Returns:
            Emoji string for the class
        """
        # Default emoji mappings - can be extended for new classes
        emoji_map = {
            "truck": "🚚",
            "tank": "🛡️",
            "soldier": "👤",
            "vehicle": "🚗",
            "person": "🧑",
            "aircraft": "✈️",
            "helicopter": "🚁",
            "ship": "🚢",
            "building": "🏢",
            "weapon": "🔫",
        }
        return emoji_map.get(obj_class.lower(), "📦")  # Default to box emoji

    def _update_static_paths(self):
        """
        Dynamically update Gradio static paths to include all collection video directories.
        This should be called whenever a new collection is created or activated.
        """
        try:
            from pathlib import Path
            import gradio as gr

            static_paths = []

            # Register all collection video directories
            collections_base = Path("./data/local_videodb/collections").absolute()
            if collections_base.exists():
                for collection_dir in collections_base.iterdir():
                    if collection_dir.is_dir():
                        videos_dir = collection_dir / "videos"
                        if videos_dir.exists():
                            static_paths.append(videos_dir)

            # Update Gradio static paths
            if static_paths:
                gr.set_static_paths(paths=static_paths)
                print(f"[Static Paths] Updated {len(static_paths)} static paths for file serving")

        except Exception as e:
            print(f"[Static Paths] Error updating static paths: {e}")
            import traceback
            traceback.print_exc()

    def process_video_upload_from_sidebar(self, video_file, segment_duration: int) -> Tuple[str, str, str]:
        """
        Process uploaded video using first active collection from sidebar

        Args:
            video_file: Uploaded video file
            segment_duration: Duration for each segment

        Returns:
            Tuple of (status_message, video_id, summary_json)
        """
        try:
            if video_file is None:
                return "❌ 업로드된 영상 파일이 없습니다", "", ""

            # Get first active collection from sidebar
            active_collections = self.collection_state.get_active_collections()
            if not active_collections:
                return "❌ 활성 컬렉션이 없습니다. 사이드바에서 최소 하나의 컬렉션을 활성화하세요.", "", ""

            target_collection = active_collections[0]
            return self.process_video_upload(video_file, segment_duration, target_collection)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", "", ""

    def process_video_upload(self, video_file, segment_duration: int, collection_id: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Process uploaded video

        Args:
            video_file: Uploaded video file
            segment_duration: Duration for each segment
            collection_id: Target collection (optional, uses current upload collection if not provided)

        Returns:
            Tuple of (status_message, video_id, summary_json)
        """
        try:
            if video_file is None:
                return "❌ 업로드된 영상 파일이 없습니다", "", ""

            # Determine target collection
            target_collection = collection_id or self.collection_state.get_upload_collection()

            # Validate collection exists
            if not self.collection_manager.validate_collection_exists(target_collection):
                return f"❌ 컬렉션 '{target_collection}'이 존재하지 않습니다", "", ""

            # Initialize or switch video system collection
            if self.video_system is None:
                # First time initialization - pass pre-loaded models to avoid reloading
                if not self.model_manager.check_models_ready():
                    raise RuntimeError("Models not loaded. Please call load_all_models() first.")
                print(f"Initializing Video Analysis System for collection: {target_collection}")
                from core_src.video_analysis_system import VideoAnalysisSystem

                # Create video system with pre-loaded models
                self.video_system = VideoAnalysisSystem(
                    collection_name=target_collection,
                    detector=self.model_manager.object_detector,
                    embedding_generator=self.model_manager.embedding_generator,
                    description_generator=self.model_manager.description_generator
                )
            else:
                # System already exists - check if we need to switch collections
                current_collection = self.video_system.videodb_manager.collection._collection_id
                if current_collection != target_collection:
                    print(f"Switching from collection '{current_collection}' to '{target_collection}'")
                    self.video_system.switch_collection(target_collection)

            # Get video path
            video_path = video_file

            # Analyze video
            status_msg = f"🔄 Processing video into collection '{target_collection}'... This may take several minutes."

            results = self.video_system.analyze_video(
                video_path=video_path,
                segment_duration=segment_duration
            )

            # Store video ID
            self.current_video_id = results["video_id"]

            # Add collection to active if not already there
            self.collection_state.add_to_active(target_collection)

            # Format summary
            summary = results["summary"]

            # Build dynamic object counts section
            object_counts_text = ""
            for obj_class, count in summary['object_counts'].items():
                # Use generic emoji for dynamic classes
                emoji = self._get_class_emoji(obj_class)
                object_counts_text += f"- {emoji} {obj_class.capitalize()}s: {count}\n"

            summary_text = f"""
✅ **영상 분석 완료!**

📁 **컬렉션:** `{target_collection}`
📹 **영상 ID:** `{results['video_id']}`
⏱️ **길이:** {summary['video_length']:.1f}초
📊 **세그먼트:** {summary['num_segments']}개

🎯 **탐지된 객체:**
{object_counts_text}
🔗 **스트림 URL:** [영상 보기]({summary['stream_url']})
"""

            summary_json = json.dumps(results, indent=2, default=str)

            return summary_text, results["video_id"], summary_json

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"❌ 영상 처리 오류: {str(e)}"
            return error_msg, "", ""

    def add_pdf_to_system(self, pdf_file) -> str:
        """
        Add PDF to RAG system

        Args:
            pdf_file: Uploaded PDF file

        Returns:
            Status message
        """
        try:
            if pdf_file is None:
                return "❌ 업로드된 PDF 파일이 없습니다"

            # Initialize agent (to access RAG system)
            self._init_agent()

            # Add PDF
            pdf_path = pdf_file
            self.agent.add_pdf_context(pdf_path)

            return f"✅ PDF가 지식 베이스에 추가되었습니다: {Path(pdf_path).name}"

        except Exception as e:
            return f"❌ PDF 추가 오류: {str(e)}"

    def refresh_collection_dropdown(self) -> gr.Dropdown:
        """
        Refresh collection dropdown choices

        Returns:
            Updated Gradio Dropdown component
        """
        try:
            choices = self.collection_manager.get_collection_choices_for_dropdown()
            current_upload = self.collection_state.get_upload_collection()

            # Find if current value is still valid
            valid_ids = [choice[1] for choice in choices]
            value = current_upload if current_upload in valid_ids else (valid_ids[0] if valid_ids else "")

            return gr.Dropdown(choices=choices, value=value)

        except Exception as e:
            print(f"Error refreshing collection dropdown: {e}")
            return gr.Dropdown(choices=[], value="")

    def refresh_active_collections_checkbox(self) -> Tuple[gr.CheckboxGroup, str]:
        """
        Refresh active collections checkbox group

        Returns:
            Tuple of (updated checkbox group, status message)
        """
        try:
            choices = self.collection_manager.get_collection_choices_for_checkbox()
            current_active = self.collection_state.get_active_collections()

            status = f"✅ 컬렉션 {len(choices)}개를 찾았습니다"
            return gr.CheckboxGroup(choices=choices, value=current_active), status

        except Exception as e:
            return gr.CheckboxGroup(choices=[], value=[]), f"❌ 오류: {str(e)}"

    def create_new_collection(self, collection_name: str, collection_description: str = "") -> Tuple[str, gr.Dropdown]:
        """
        Create a new collection

        Args:
            collection_name: Name for the new collection
            collection_description: Optional description

        Returns:
            Tuple of (status message, updated dropdown)
        """
        try:
            success, message, collection_id = self.collection_manager.create_collection(
                name=collection_name,
                description=collection_description
            )

            if success and collection_id:
                # Add to active collections
                self.collection_state.add_to_active(collection_id)
                # Set as upload target
                self.collection_state.set_upload_collection(collection_id)

            # Refresh dropdown
            dropdown = self.refresh_collection_dropdown()

            return message, dropdown

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 오류: {str(e)}", gr.Dropdown()

    def delete_collection_from_sidebar(self, collection_id: str, force_delete: bool) -> Tuple[str, gr.Dropdown]:
        """
        Delete a collection from the sidebar

        Args:
            collection_id: Collection ID to delete
            force_delete: If True, delete even if collection has videos

        Returns:
            Tuple of (status message, updated dropdown)
        """
        try:
            if not collection_id:
                return "⚠️ 삭제할 컬렉션을 선택하세요", gr.Dropdown()

            # Confirm deletion details
            collection_info = self.collection_manager.get_collection_info(collection_id)
            if collection_info:
                video_count = collection_info.get("video_count", 0)
                collection_name = collection_info.get("name", collection_id)

                if video_count > 0 and not force_delete:
                    return f"⚠️ 컬렉션 '{collection_name}'에 영상 {video_count}개가 있습니다.\n\n✅ 삭제를 확인하려면 '강제 삭제'를 선택하세요.", gr.Dropdown()

            # Delete collection
            success, message = self.collection_manager.delete_collection(
                collection_id=collection_id,
                force=force_delete
            )

            if success:
                # Remove from active collections if present
                active_collections = self.collection_state.get_active_collections()
                if collection_id in active_collections:
                    active_collections.remove(collection_id)
                    self.collection_state.set_active_collections(active_collections)
                    set_active_collections(active_collections)

                # Clear upload target if it was this collection
                if self.collection_state.get_upload_collection() == collection_id:
                    # Set to first available collection or empty
                    available = self.collection_manager.get_collection_choices_for_dropdown()
                    new_upload = available[0][1] if available else ""
                    self.collection_state.set_upload_collection(new_upload)

                # Clear selected videos if they were from this collection
                if hasattr(self, 'selected_video_ids') and self.selected_video_ids:
                    # Filter out videos from deleted collection
                    # Note: We don't have collection info anymore, so clear all selections to be safe
                    self.selected_video_ids = []
                    set_selected_video_ids([])

                print(f"[Collection Manager] Successfully deleted collection: {collection_id}")

            # Refresh dropdown
            dropdown = self.refresh_collection_dropdown()

            return f"✅ {message}" if success else f"❌ {message}", dropdown

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 컬렉션 삭제 오류: {str(e)}", gr.Dropdown()

    def create_new_collection_global(self, collection_name: str, collection_description: str = "") -> Tuple[str, gr.CheckboxGroup]:
        """
        Create a new collection for global collection manager

        Args:
            collection_name: Name for the new collection
            collection_description: Optional description

        Returns:
            Tuple of (status message, updated checkbox group)
        """
        try:
            success, message, collection_id = self.collection_manager.create_collection(
                name=collection_name,
                description=collection_description
            )

            if success and collection_id:
                # Add to active collections
                self.collection_state.add_to_active(collection_id)
                # Notify tools about updated active collections
                set_active_collections(self.collection_state.get_active_collections())

                # Update static paths to include new collection's video directory
                print(f"[Collection Manager] Updating static paths for new collection: {collection_id}")
                self._update_static_paths()

            # Refresh checkbox choices and values
            checkbox = gr.CheckboxGroup(
                choices=self.collection_manager.get_collection_choices_for_checkbox(),
                value=self.collection_state.get_active_collections()
            )

            return message, checkbox

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 오류: {str(e)}", gr.CheckboxGroup()

    def update_active_collection_single(self, selected_collection: str) -> Tuple[str, gr.HTML, str, str]:
        """
        Update active collection from dropdown (single selection) and auto-refresh video contexts

        Args:
            selected_collection: Selected collection ID from dropdown

        Returns:
            Tuple of (status message, video thumbnails HTML, selection state JSON, selection status)
        """
        try:
            if not selected_collection:
                # Clear everything if no collection selected
                self.collection_state.set_active_collections([])
                set_active_collections([])

                # Clear selected videos
                self.selected_video_ids = []
                set_selected_video_ids([])

                empty_html = "<p style='text-align: center; color: #666; padding: 40px;'>선택된 컬렉션이 없습니다.</p>"
                return "ℹ️ 활성 컬렉션 없음", empty_html, "[]", "선택된 컬렉션 없음"

            # Update to single active collection
            self.collection_state.set_active_collections([selected_collection])
            set_active_collections([selected_collection])

            # Update static paths to ensure collection's video directory is registered
            print(f"[Collection Change] Updating static paths for collection: {selected_collection}")
            self._update_static_paths()

            # Clear previously selected video IDs from old collection
            print(f"[Collection Change] Clearing previous video selections: {self.selected_video_ids}")
            self.selected_video_ids = []
            set_selected_video_ids([])

            # Auto-refresh video contexts for new collection
            print(f"[Collection Change] Auto-refreshing video contexts for collection: {selected_collection}")
            thumbnails_html, selection_state, refresh_status = self.refresh_video_contexts()

            status_msg = f"✅ 활성화됨: {selected_collection}\n🔄 영상 새로고침 완료. 이전 선택이 초기화되었습니다."

            return status_msg, thumbnails_html, selection_state, refresh_status

        except Exception as e:
            import traceback
            traceback.print_exc()
            empty_html = f"<p style='color: #f44336;'>오류: {str(e)}</p>"
            return f"❌ 오류: {str(e)}", empty_html, "[]", f"오류: {str(e)}"

    def update_active_collections(self, selected_collections: list) -> str:
        """
        Update which collections are active for context scanning

        Args:
            selected_collections: List of selected collection IDs

        Returns:
            Status message
        """
        try:
            self.collection_state.set_active_collections(selected_collections)

            # Notify the tools about active collections
            set_active_collections(selected_collections)

            if not selected_collections:
                return "ℹ️ 활성 컬렉션이 없습니다. 에이전트가 영상 컨텍스트에 접근할 수 없습니다."

            return f"✅ 컬렉션 {len(selected_collections)}개 활성화됨: {', '.join(selected_collections)}"

        except Exception as e:
            return f"❌ 활성 컬렉션 업데이트 오류: {str(e)}"

    def update_upload_collection(self, collection_id: str) -> str:
        """
        Update the target collection for video uploads

        Args:
            collection_id: Collection ID to upload to

        Returns:
            Status message
        """
        try:
            if not collection_id:
                return "❌ 선택된 컬렉션이 없습니다"

            self.collection_state.set_upload_collection(collection_id)
            return f"✅ 영상이 컬렉션에 업로드됩니다: {collection_id}"

        except Exception as e:
            return f"❌ 업로드 컬렉션 업데이트 오류: {str(e)}"

    def refresh_video_contexts(self):
        """
        Refresh list of available video contexts from active collections

        Returns:
            Tuple of (interactive thumbnail HTML, selected IDs, status message)
        """
        try:
            import base64
            import json

            # Get active collections
            active_collections = self.collection_state.get_active_collections()

            if not active_collections:
                empty_html = "<p style='text-align: center; color: #666; padding: 40px;'>활성 컬렉션이 없습니다. 컬렉션 관리자 섹션에서 컬렉션을 활성화하세요.</p>"
                return empty_html, json.dumps([]), "활성 컬렉션이 없습니다. 영상을 보려면 컬렉션을 활성화하세요."

            # Scan videos from all active collections
            videos = self.context_scanner.scan_available_videos(collection_ids=active_collections)

            if not videos:
                empty_html = "<p style='text-align: center; color: #666; padding: 40px;'>영상을 찾을 수 없습니다. 먼저 영상을 업로드하고 분석하세요.</p>"
                return empty_html, json.dumps([]), "영상을 찾을 수 없습니다. 먼저 영상을 업로드하고 분석하세요."

            # Build interactive thumbnail grid with JavaScript
            thumbnail_html = """
            <div id="video-selector-container">
                <style>
                    .video-card {
                        border: 3px solid #ddd;
                        border-radius: 12px;
                        padding: 12px;
                        background: white;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        cursor: pointer;
                        transition: all 0.3s ease;
                        position: relative;
                    }
                    .video-card:hover {
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        transform: translateY(-2px);
                    }
                    .video-card.selected {
                        border-color: #4CAF50;
                        background: #f0f9f0;
                        box-shadow: 0 4px 12px rgba(76,175,80,0.3);
                    }
                    .video-card .selection-indicator {
                        position: absolute;
                        top: 8px;
                        right: 8px;
                        width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background: white;
                        border: 2px solid #ddd;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 18px;
                        transition: all 0.3s ease;
                    }
                    .video-card.selected .selection-indicator {
                        background: #4CAF50;
                        border-color: #4CAF50;
                        color: white;
                    }
                    .video-card .delete-btn {
                        position: absolute;
                        top: 8px;
                        left: 8px;
                        width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background: white;
                        border: 2px solid #f44336;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 16px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        z-index: 10;
                    }
                    .video-card .delete-btn:hover {
                        background: #f44336;
                        color: white;
                        transform: scale(1.1);
                    }
                    .video-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                        gap: 15px;
                        padding: 10px;
                    }
                </style>
                <div class="video-grid">
            """

            # Store video data for JavaScript
            video_data = []

            for video in videos:
                video_id = video['video_id']
                video_name = video['name']
                num_segments = video['num_segments']
                num_objects = video.get('total_objects_detected', 0)

                # Store for JavaScript
                video_data.append({
                    'id': video_id,
                    'name': video_name,
                    'segments': num_segments,
                    'objects': num_objects
                })

                # Create thumbnail card
                thumbnail_path = video.get('thumbnail_path', '')

                # Handle thumbnail display
                if thumbnail_path and Path(thumbnail_path).exists():
                    try:
                        with open(thumbnail_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            ext = Path(thumbnail_path).suffix.lower()
                            img_format = 'jpeg' if ext in ['.jpg', '.jpeg'] else 'png'
                            img_display = f"<img src='data:image/{img_format};base64,{img_data}' style='width: 100%; height: 130px; object-fit: cover; border-radius: 8px;'/>"
                    except Exception as e:
                        print(f"Error loading thumbnail {thumbnail_path}: {e}")
                        img_display = "<div style='width: 100%; height: 130px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;'>🎬</div>"
                else:
                    img_display = "<div style='width: 100%; height: 130px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;'>🎬</div>"

                # Check if currently selected
                is_selected = video_id in self.selected_video_ids
                selected_class = " selected" if is_selected else ""
                check_icon = "✓" if is_selected else ""

                # Create clickable card
                # Escape video_name for JavaScript string
                video_name_escaped = video_name.replace("'", "\\'").replace('"', '\\"')

                thumbnail_html += f"""
                <div class='video-card{selected_class}' onclick='toggleVideoSelection("{video_id}")' data-video-id='{video_id}'>
                    <div class='delete-btn' onclick='(function(e){{e.stopPropagation(); deleteVideoContext("{video_id}", "{video_name_escaped}"); return false;}})(event); return false;' title='Delete video'>🗑️</div>
                    <div class='selection-indicator'>{check_icon}</div>
                    {img_display}
                    <div style='margin-top: 10px; font-size: 13px; font-weight: bold; color: #333; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;' title='{video_name}'>{video_name}</div>
                    <div style='font-size: 11px; color: #666; margin-top: 6px; line-height: 1.4;'>
                        📊 {num_segments} segments<br/>
                        🎯 {num_objects} objects<br/>
                        🆔 {video_id[:12]}...
                    </div>
                </div>
                """

            thumbnail_html += """
                </div>
                <!-- Initialize selection state via onload event -->
                <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
                     onload="initializeSelection('""" + json.dumps(self.selected_video_ids).replace("'", "\\'") + """')"
                     style="display:none;"
                     alt="" />
            </div>
            """

            status = f"✅ 분석된 영상 {len(videos)}개를 찾았습니다. 썸네일을 클릭하여 선택하세요."
            return thumbnail_html, json.dumps(list(self.selected_video_ids)), status

        except Exception as e:
            error_html = f"<p style='color: red; padding: 20px;'>❌ 오류: {str(e)}</p>"
            return error_html, json.dumps([]), f"❌ 영상 스캔 오류: {str(e)}"

    def refresh_pdf_contexts(self):
        """
        Refresh list of available PDF contexts

        Returns:
            Tuple of (CheckboxGroup choices, status message)
        """
        try:
            pdfs = self.context_scanner.scan_available_pdfs()

            if not pdfs:
                return gr.CheckboxGroup(choices=[], value=[]), "PDF를 찾을 수 없습니다. 먼저 지식 베이스에 PDF를 추가하세요."

            # Create choices for CheckboxGroup
            # Format: (label, value) where value is filename
            choices = []
            for pdf in pdfs:
                label = f"{pdf['filename']} ({pdf['num_chunks']}개 청크)"
                choices.append((label, pdf['filename']))

            status = f"✅ 인덱싱된 PDF {len(pdfs)}개를 찾았습니다"
            return gr.CheckboxGroup(choices=choices, value=self.selected_pdf_files), status

        except Exception as e:
            return gr.CheckboxGroup(choices=[], value=[]), f"❌ PDF 스캔 오류: {str(e)}"

    def update_selected_videos(self, selected_video_ids):
        """
        Update selected video contexts

        Args:
            selected_video_ids: List of selected video IDs

        Returns:
            Status message
        """
        self.selected_video_ids = selected_video_ids if selected_video_ids else []

        # Update tool global state
        set_selected_video_ids(self.selected_video_ids)

        if not self.selected_video_ids:
            return "ℹ️ 선택된 영상이 없습니다. 에이전트가 영상 컨텍스트를 사용할 수 없습니다."

        # Also update current_video_id for backward compatibility
        if self.selected_video_ids:
            self.current_video_id = self.selected_video_ids[0]

        return f"✅ 영상 {len(self.selected_video_ids)}개 선택됨: {', '.join(self.selected_video_ids)}"

    def parse_video_selection(self, json_state):
        """
        Parse JSON selection state from JavaScript and validate against available videos

        Args:
            json_state: JSON string of selected video IDs

        Returns:
            Status message
        """
        try:
            import json
            print(f"[Backend] Received video selection state: {json_state}")
            selected_ids = json.loads(json_state) if json_state else []

            # Validate selected IDs against available videos in active collections
            active_collections = self.collection_state.get_active_collections()
            available_videos = self.context_scanner.scan_available_videos(collection_ids=active_collections)
            available_video_ids = set(video['video_id'] for video in available_videos)

            # Filter out any video IDs that no longer exist
            valid_selected_ids = [vid for vid in selected_ids if vid in available_video_ids]

            # Check if any IDs were filtered out
            removed_count = len(selected_ids) - len(valid_selected_ids)
            if removed_count > 0:
                print(f"[Backend] Removed {removed_count} invalid/deleted video ID(s) from selection")

            self.selected_video_ids = valid_selected_ids

            # Update tool global state
            set_selected_video_ids(self.selected_video_ids)

            print(f"[Backend] Updated selected_video_ids: {self.selected_video_ids}")

            if not self.selected_video_ids:
                if removed_count > 0:
                    return f"ℹ️ 삭제된 영상 {removed_count}개가 선택에서 제거되었습니다. 현재 선택된 영상이 없습니다."
                return "ℹ️ 선택된 영상이 없습니다. 썸네일을 클릭하여 선택하세요."

            # Also update current_video_id for backward compatibility
            if self.selected_video_ids:
                self.current_video_id = self.selected_video_ids[0]

            status_msg = f"✅ 영상 {len(self.selected_video_ids)}개 선택됨: {', '.join([vid[:12] + '...' for vid in self.selected_video_ids])}"
            if removed_count > 0:
                status_msg += f" (삭제된 영상 {removed_count}개 제거됨)"

            return status_msg

        except Exception as e:
            print(f"[Backend] Error parsing video selection: {e}")
            import traceback
            traceback.print_exc()
            return "❌ 선택 파싱 오류"

    def update_selected_pdfs(self, selected_pdf_files):
        """
        Update selected PDF contexts

        Args:
            selected_pdf_files: List of selected PDF filenames

        Returns:
            Status message
        """
        self.selected_pdf_files = selected_pdf_files if selected_pdf_files else []

        # Update tool global state
        set_selected_pdf_files(self.selected_pdf_files)

        if not self.selected_pdf_files:
            return "ℹ️ 선택된 PDF가 없습니다. 에이전트가 PDF 컨텍스트를 사용할 수 없습니다."

        return f"✅ PDF {len(self.selected_pdf_files)}개 선택됨: {', '.join(self.selected_pdf_files)}"

    def delete_video_context(self, video_id: str):
        """
        Delete a video context from the system.
        Removes video files, metadata, segments, FAISS index, and tracking videos.

        Args:
            video_id: Video ID to delete

        Returns:
            Status message
        """
        try:
            import shutil
            from pathlib import Path

            if not video_id:
                return ""  # Return empty string for empty input (triggered by field clearing)

            # Strip whitespace
            video_id = video_id.strip()
            if not video_id:
                return ""

            # Find which collection this video belongs to
            collection_id = None
            active_collections = self.collection_state.get_active_collections()

            # Search through active collections to find the video
            for coll_id in active_collections:
                video_path = Path(f"./data/local_videodb/collections/{coll_id}/videos/{video_id}")
                if video_path.exists():
                    collection_id = coll_id
                    break

            if not collection_id:
                return f"❌ 오류: 영상 {video_id}를 활성 컬렉션에서 찾을 수 없습니다"

            # Get video directory path
            video_path = Path(f"./data/local_videodb/collections/{collection_id}/videos/{video_id}")

            # Remove from selected list if present
            if video_id in self.selected_video_ids:
                self.selected_video_ids.remove(video_id)
                set_selected_video_ids(self.selected_video_ids)

            # Delete entire video directory (includes original video, metadata, segments, frames, tracking, FAISS)
            shutil.rmtree(video_path)

            # Update collection metadata.json to remove video_id from videos list
            collection_metadata_path = Path(f"./data/local_videodb/collections/{collection_id}/metadata.json")
            if collection_metadata_path.exists():
                try:
                    with open(collection_metadata_path, 'r') as f:
                        collection_metadata = json.load(f)

                    # Remove video_id from videos list if present
                    if "videos" in collection_metadata and video_id in collection_metadata["videos"]:
                        collection_metadata["videos"].remove(video_id)

                        # Save updated metadata
                        with open(collection_metadata_path, 'w') as f:
                            json.dump(collection_metadata, f, indent=2)

                        print(f"Removed {video_id} from collection {collection_id} metadata")
                except Exception as meta_error:
                    print(f"Warning: Could not update collection metadata: {meta_error}")
                    # Don't fail the deletion if metadata update fails

            return f"✅ 컬렉션 {collection_id}에서 영상 {video_id} 삭제 완료"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 영상 삭제 오류: {str(e)}"

    def delete_pdf_context(self, pdf_filename: str):
        """
        Delete a PDF context from the RAG system.
        Removes PDF file, metadata, and updates FAISS index.

        Args:
            pdf_filename: PDF filename to delete

        Returns:
            Status message
        """
        try:
            if not pdf_filename:
                return ""  # Return empty string for empty input (triggered by field clearing)

            # Strip whitespace
            pdf_filename = pdf_filename.strip()
            if not pdf_filename:
                return ""

            # Get RAG system
            rag_system = get_rag_system()

            # Remove PDF using existing remove_pdf method
            success = rag_system.remove_pdf(pdf_filename)

            if success:
                # Remove from selected list if present
                if pdf_filename in self.selected_pdf_files:
                    self.selected_pdf_files.remove(pdf_filename)
                    set_selected_pdf_files(self.selected_pdf_files)

                return f"✅ PDF 삭제 완료: {pdf_filename}"
            else:
                return f"❌ PDF 삭제 실패: {pdf_filename}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ PDF 삭제 오류: {str(e)}"

    def delete_selected_pdfs(self, selected_pdfs: list):
        """
        Delete multiple selected PDFs from the RAG system.

        Args:
            selected_pdfs: List of selected PDF filenames

        Returns:
            Status message
        """
        try:
            if not selected_pdfs:
                return "⚠️ 삭제할 PDF가 선택되지 않았습니다"

            # Get RAG system
            rag_system = get_rag_system()

            deleted_count = 0
            failed_count = 0
            errors = []

            for pdf_filename in selected_pdfs:
                try:
                    success = rag_system.remove_pdf(pdf_filename)
                    if success:
                        deleted_count += 1
                        # Remove from selected list if present
                        if pdf_filename in self.selected_pdf_files:
                            self.selected_pdf_files.remove(pdf_filename)
                    else:
                        failed_count += 1
                        errors.append(f"{pdf_filename} 삭제 실패")
                except Exception as e:
                    failed_count += 1
                    errors.append(f"{pdf_filename} 삭제 오류: {str(e)}")

            # Update global state
            set_selected_pdf_files(self.selected_pdf_files)

            # Build status message
            status = f"✅ PDF {deleted_count}개 삭제 완료"
            if failed_count > 0:
                status += f"\n❌ PDF {failed_count}개 삭제 실패"
                if errors:
                    status += "\n\n오류:\n" + "\n".join(errors[:5])  # Show first 5 errors

            return status

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ PDF 삭제 오류: {str(e)}"

    def get_tracking_videos(self):
        """
        Retrieve all tracking videos from active collections organized by video_id and segment.

        Returns:
            List of dicts containing video_id and tracking video information
        """
        try:
            from pathlib import Path
            from core_src.videodb_manager import VideoDBManager

            # Get active collections
            active_collections = self.collection_state.get_active_collections()
            print(f"\n{'='*60}")
            print(f"[Tracking Videos] Loading tracking videos...")
            print(f"[Tracking Videos] Active collections: {active_collections}")

            if not active_collections:
                print("[Tracking Videos] ❌ No active collections selected")
                return []

            # Get videos from active collections only
            videos = self.context_scanner.scan_available_videos(collection_ids=active_collections)
            print(f"[Tracking Videos] Found {len(videos)} videos in active collections")

            if not videos:
                print("[Tracking Videos] ℹ️ No videos found in active collections")
                return []

            tracking_videos_data = []

            # Create manager dictionary for each collection
            managers = {}

            for video in videos:
                video_id = video['video_id']
                video_name = video['name']
                collection_id = video.get('collection_id', 'battlefield_reconnaissance')

                print(f"\n[Tracking Videos] Processing video: {video_name}")
                print(f"  - Video ID: {video_id}")
                print(f"  - Collection ID: {collection_id}")

                # Get tracking directory for this video using its collection_id
                tracking_dir = Path(f"./data/local_videodb/collections/{collection_id}/videos/{video_id}/tracking")
                print(f"  - Tracking dir: {tracking_dir}")
                print(f"  - Dir exists: {tracking_dir.exists()}")

                if not tracking_dir.exists():
                    print(f"  - ⚠️ Tracking directory not found, skipping")
                    continue

                # Get all tracking videos in this directory
                tracking_files = sorted(tracking_dir.glob("segment_*_tracked.mp4"))
                print(f"  - Found {len(tracking_files)} tracking files")

                if not tracking_files:
                    print(f"  - ℹ️ No tracking files in directory, skipping")
                    continue

                # Get or create VideoDBManager for this collection
                if collection_id not in managers:
                    managers[collection_id] = VideoDBManager(
                        storage_path="./data/local_videodb",
                        collection_name=collection_id
                    )

                manager = managers[collection_id]

                # Collect segment info
                segments_info = []
                for track_file in tracking_files:
                    # Extract segment_id from filename (segment_000_tracked.mp4 -> 0)
                    try:
                        segment_id = int(track_file.stem.split('_')[1])
                    except (IndexError, ValueError):
                        segment_id = None

                    # Get metadata for this segment
                    metadata = None
                    if segment_id is not None:
                        try:
                            metadata = manager.get_segment_metadata(video_id, segment_id)
                        except Exception as e:
                            print(f"Error getting metadata for segment {segment_id} in collection {collection_id}: {e}")

                    # Build segment info
                    segment_data = {
                        'segment_id': segment_id,
                        'tracking_path': str(track_file),
                        'filename': track_file.name,
                        'file_size_mb': track_file.stat().st_size / (1024 * 1024),
                    }

                    # Add metadata if available
                    if metadata:
                        segment_data['start_time'] = metadata.get('start_time', 0)
                        segment_data['end_time'] = metadata.get('end_time', 0)
                        segment_data['num_objects_tracked'] = metadata.get('num_objects_tracked', 0)
                        segment_data['tracking_enabled'] = metadata.get('tracking_enabled', False)
                        segment_data['event_description'] = metadata.get('event_description', '')

                    segments_info.append(segment_data)

                # Add to result
                tracking_videos_data.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'collection_id': collection_id,
                    'num_tracking_videos': len(segments_info),
                    'segments': segments_info
                })

            # Cleanup managers
            for manager in managers.values():
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()

            # Log summary
            print(f"\n[Tracking Videos] Summary:")
            print(f"  - Total videos with tracking: {len(tracking_videos_data)}")
            total_segments = sum(v['num_tracking_videos'] for v in tracking_videos_data)
            print(f"  - Total tracking segments: {total_segments}")
            for v in tracking_videos_data:
                print(f"  - {v['video_name']} ({v['collection_id']}): {v['num_tracking_videos']} segments")
            print(f"{'='*60}\n")

            return tracking_videos_data

        except Exception as e:
            print(f"Error retrieving tracking videos: {e}")
            import traceback
            traceback.print_exc()
            return []

    def render_tracking_gallery(self):
        """
        Render tracking videos in a gallery format with rows per video.

        Returns:
            Tuple of (HTML string, status message)
        """
        try:
            print(f"\n[Render Gallery] Starting to render tracking gallery...")
            tracking_data = self.get_tracking_videos()
            print(f"[Render Gallery] Received {len(tracking_data)} videos with tracking data")

            if not tracking_data:
                empty_html = """
                <div style='text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;'>
                    <div style='font-size: 64px; margin-bottom: 20px;'>🎬</div>
                    <h2 style='margin: 0 0 10px 0;'>추적 영상을 찾을 수 없습니다</h2>
                    <p style='margin: 0; opacity: 0.9;'>SAM3 추적이 활성화된 영상을 업로드하고 분석하면 여기에 추적 영상이 표시됩니다.</p>
                </div>
                """
                return empty_html, "추적 영상을 찾을 수 없습니다. 먼저 추적이 활성화된 영상을 분석하세요."

            # Build HTML gallery
            gallery_html = """
            <style>
                .tracking-gallery {
                    padding: 20px 0;
                }
                .video-row {
                    margin-bottom: 40px;
                    background: white;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                .video-row-header {
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #e0e0e0;
                }
                .video-row-title {
                    font-size: 22px;
                    font-weight: bold;
                    color: #1a1a1a;
                    margin: 0 0 10px 0;
                }
                .video-row-info {
                    font-size: 15px;
                    color: #2d2d2d;
                    margin: 0;
                    font-weight: 500;
                }
                .segments-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                    gap: 20px;
                }
                .segment-card {
                    background: #f9f9f9;
                    border-radius: 10px;
                    overflow: hidden;
                    border: 2px solid #e0e0e0;
                    transition: all 0.3s ease;
                }
                .segment-card:hover {
                    border-color: #667eea;
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
                    transform: translateY(-2px);
                }
                .segment-video {
                    width: 100%;
                    height: 200px;
                    background: #000;
                    object-fit: contain;
                    display: block;
                }
                .segment-info {
                    padding: 12px;
                    background: white;
                }
                .segment-title {
                    font-size: 16px;
                    font-weight: bold;
                    color: #1a1a1a;
                    margin: 0 0 10px 0;
                }
                .segment-meta {
                    font-size: 13px;
                    color: #2d2d2d;
                    line-height: 1.8;
                    font-weight: 500;
                }
                .segment-meta-item {
                    margin: 4px 0;
                }
                .event-desc {
                    font-size: 13px;
                    color: #1a1a1a;
                    margin-top: 10px;
                    padding: 10px;
                    background: #f8f9ff;
                    border-radius: 6px;
                    font-style: italic;
                    border-left: 3px solid #667eea;
                    line-height: 1.6;
                }
            </style>
            <div class="tracking-gallery">
            """

            total_tracking_videos = 0

            # Render each video's tracking footage in a row
            for video_data in tracking_data:
                video_id = video_data['video_id']
                video_name = video_data['video_name']
                collection_id = video_data.get('collection_id', 'unknown')
                num_tracking = video_data['num_tracking_videos']
                segments = video_data['segments']

                total_tracking_videos += num_tracking

                # Video row header
                gallery_html += f"""
                <div class="video-row">
                    <div class="video-row-header">
                        <h3 class="video-row-title">🎥 {video_name}</h3>
                        <p class="video-row-info">
                            📂 Collection: <strong>{collection_id}</strong>  |  📊 {num_tracking} tracking segments  |  🆔 {video_id}
                        </p>
                    </div>
                    <div class="segments-grid">
                """

                # Render each segment's tracking video
                for segment in segments:
                    segment_id = segment.get('segment_id', '?')
                    tracking_path = segment['tracking_path']
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', 0)
                    num_objects = segment.get('num_objects_tracked', 0)
                    file_size = segment['file_size_mb']
                    event_desc = segment.get('event_description', '')

                    # Convert to absolute path for Gradio
                    from pathlib import Path
                    abs_tracking_path = Path(tracking_path).absolute()
                    print(f"[Render Gallery] Segment #{segment_id} ({collection_id})")
                    print(f"  - Relative path: {tracking_path}")
                    print(f"  - Absolute path: {abs_tracking_path}")
                    print(f"  - File exists: {abs_tracking_path.exists()}")

                    # Verify file exists
                    if not abs_tracking_path.exists():
                        print(f"  - ❌ Warning: Tracking video not found, skipping")
                        continue

                    # Format time range
                    time_range = f"{start_time:.1f}s - {end_time:.1f}s"
                    duration = end_time - start_time if end_time > start_time else 0

                    # Create video element with proper attributes for playback
                    gallery_html += f"""
                    <div class="segment-card">
                        <video
                            class="segment-video"
                            controls
                            preload="metadata"
                            playsinline
                            webkit-playsinline
                        >
                            <source src="/gradio_api/file={abs_tracking_path}" type="video/mp4">
                            <source src="/file={abs_tracking_path}" type="video/mp4">
                            <p>Your browser does not support HTML5 video.
                               <a href="/gradio_api/file={abs_tracking_path}" target="_blank">Download video</a>
                            </p>
                        </video>
                        <div class="segment-info">
                            <div class="segment-title">📹 Segment #{segment_id}</div>
                            <div class="segment-meta">
                                <div class="segment-meta-item">⏱️ {time_range} ({duration:.1f}s)</div>
                                <div class="segment-meta-item">🎯 {num_objects} objects tracked</div>
                                <div class="segment-meta-item">💾 {file_size:.2f} MB</div>
                            </div>
                    """

                    # Add event description if available
                    if event_desc:
                        gallery_html += f"""
                            <div class="event-desc">
                                💭 {event_desc}
                            </div>
                        """

                    gallery_html += """
                        </div>
                    </div>
                    """

                # Close video row
                gallery_html += """
                    </div>
                </div>
                """

            # Close gallery
            gallery_html += "</div>"

            # Add JavaScript for video error handling and debugging
            gallery_html += """
            <script>
            (function() {
                console.log('[Tracking Gallery] Initializing video players...');

                // Wait for DOM to be ready
                setTimeout(function() {
                    const videos = document.querySelectorAll('.tracking-gallery video');
                    console.log('[Tracking Gallery] Found ' + videos.length + ' video elements');

                    videos.forEach(function(video, index) {
                        // Log video source
                        const sources = video.querySelectorAll('source');
                        sources.forEach(function(source) {
                            console.log('[Video ' + index + '] Source:', source.src);
                        });

                        // Add error handler
                        video.addEventListener('error', function(e) {
                            console.error('[Video ' + index + '] Error loading video:', e);
                            console.error('[Video ' + index + '] Video error code:', video.error ? video.error.code : 'unknown');
                            console.error('[Video ' + index + '] Video error message:', video.error ? video.error.message : 'unknown');
                        });

                        // Add loadedmetadata event
                        video.addEventListener('loadedmetadata', function() {
                            console.log('[Video ' + index + '] Metadata loaded successfully');
                            console.log('[Video ' + index + '] Duration:', video.duration);
                        });

                        // Add canplay event
                        video.addEventListener('canplay', function() {
                            console.log('[Video ' + index + '] Can play - video is ready');
                        });

                        // Source error handlers
                        sources.forEach(function(source, sourceIndex) {
                            source.addEventListener('error', function(e) {
                                console.error('[Video ' + index + ' Source ' + sourceIndex + '] Failed to load:', source.src);
                            });
                        });

                        // Try to load the video
                        video.load();
                    });
                }, 500);
            })();
            </script>
            """

            # Get active collections for status message
            active_collections = self.collection_state.get_active_collections()
            collections_str = ', '.join(active_collections) if active_collections else '컬렉션 없음'

            status_msg = f"✅ 활성 컬렉션에서 영상 {len(tracking_data)}개의 추적 영상 {total_tracking_videos}개 표시 중: {collections_str}"

            return gallery_html, status_msg

        except Exception as e:
            error_html = f"""
            <div style='text-align: center; padding: 40px; background: #ffebee; border-radius: 12px; color: #c62828;'>
                <div style='font-size: 48px; margin-bottom: 20px;'>⚠️</div>
                <h3>추적 영상 로드 오류</h3>
                <p>{str(e)}</p>
            </div>
            """
            import traceback
            traceback.print_exc()
            return error_html, f"❌ 오류: {str(e)}"

    def _prepare_message_with_context(self, user_message: str) -> str:
        """
        Prepare enhanced message with selected context information for skill-based analysis.
        Injects context summary so the agent knows what's available before using skills.

        Args:
            user_message: Original user query

        Returns:
            Enhanced message with context information prepended
        """
        from tools.videodb_query_tool import get_selected_video_ids
        from tools.pdf_rag_tool import get_selected_pdf_files

        # Get selected contexts
        selected_videos = get_selected_video_ids()
        selected_pdfs = get_selected_pdf_files()

        # Check for tactical map state
        tactical_map_available = False
        tactical_map_summary = ""
        try:
            wargame_state_path = _get_temp_dir() / "wargame_state.json"
            if wargame_state_path.exists():
                with open(wargame_state_path, 'r') as f:
                    wargame_state = json.load(f)

                # Validate that state is a dictionary
                if not isinstance(wargame_state, dict):
                    print(f"[Context] Invalid wargame state format: expected dict, got {type(wargame_state).__name__}")
                    wargame_state = {}

                # Count units by affiliation
                all_units = []
                if "military_units" in wargame_state:
                    all_units.extend(wargame_state["military_units"])
                if "map_clicked_units" in wargame_state:
                    all_units.extend(wargame_state["map_clicked_units"])

                if all_units:
                    tactical_map_available = True
                    friendly_count = sum(1 for u in all_units if u.get("affiliation", "").upper() == "FRIEND")
                    hostile_count = sum(1 for u in all_units if u.get("affiliation", "").upper() == "HOSTILE")

                    # Get unit type breakdown
                    friendly_types = {}
                    hostile_types = {}
                    for u in all_units:
                        unit_type = u.get("unit_type", "UNKNOWN")
                        affiliation = u.get("affiliation", "").upper()
                        if affiliation == "FRIEND":
                            friendly_types[unit_type] = friendly_types.get(unit_type, 0) + 1
                        elif affiliation == "HOSTILE":
                            hostile_types[unit_type] = hostile_types.get(unit_type, 0) + 1

                    tactical_map_summary = f"{len(all_units)} units ({friendly_count} friendly, {hostile_count} hostile)"
                    if friendly_types:
                        types_str = ", ".join([f"{count} {utype}" for utype, count in friendly_types.items()])
                        tactical_map_summary += f" | Friendly: {types_str}"
                    if hostile_types:
                        types_str = ", ".join([f"{count} {utype}" for utype, count in hostile_types.items()])
                        tactical_map_summary += f" | Hostile: {types_str}"
        except Exception as e:
            print(f"[Context] Error reading tactical map state: {e}")

        # If no contexts selected at all
        if not selected_videos and not selected_pdfs and not tactical_map_available:
            return f"""⚠️ NO CONTEXTS SELECTED

The user has not selected any contexts to analyze. Inform them to:
1. Go to the Context Catalog tab to select videos and/or PDFs
2. Or add units to the tactical map
3. Return and ask their question again

Do NOT attempt to use videodb_query_skill or pdf_rag_skill - no contexts available.

---

User Query: {user_message}"""

        # Build concise context summary for skill-based analysis
        context_parts = []

        if selected_videos:
            context_parts.append(f"**Videos ({len(selected_videos)}):** {', '.join(selected_videos[:3])}{'...' if len(selected_videos) > 3 else ''}")

        if selected_pdfs:
            context_parts.append(f"**PDFs ({len(selected_pdfs)}):** {', '.join(selected_pdfs[:3])}{'...' if len(selected_pdfs) > 3 else ''}")

        if tactical_map_available:
            context_parts.append(f"**Tactical Map:** {tactical_map_summary}")

        context_summary = "\n".join(context_parts)

        return f"""📋 CONTEXT SUMMARY
{context_summary}

---

**User Query:** {user_message}

---

**Skill Usage Instructions:**
1. Start with `videodb_query_skill("get_contexts")` to confirm available contexts
2. Use appropriate skills to gather evidence:
   - `videodb_query_skill(action, ...)` for video analysis
   - `pdf_rag_skill(action, ...)` for document search
   - `wargame_query_skill(action, ...)` for tactical map data
3. Generate final answer using `final_response_skill(action, ...)` for structured report

Skills automatically use selected contexts - no need to specify IDs unless querying specific items."""

    def stop_agent(self):
        """
        Stop agent execution by calling agent.interrupt()

        Returns:
            Status message
        """
        try:
            if self.agent is not None and hasattr(self.agent, 'agent'):
                # Call interrupt() on the underlying agent
                self.agent.agent.interrupt()
                print("[Stop] Agent interrupt signal sent")
                return "⏹️ 에이전트에 중지 신호를 보냈습니다. 현재 단계가 완료될 때까지 대기 중..."
            else:
                print("[Stop] No active agent to interrupt")
                return "ℹ️ 중지할 활성 에이전트 실행이 없습니다"
        except Exception as e:
            print(f"[Stop] Error interrupting agent: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 에이전트 중지 오류: {str(e)}"

    def chat_with_agent(self, message: str, history: list):
        """
        Chat with agent with streaming display using gr.ChatMessage for collapsible tool calls.

        Args:
            message: User message
            history: Chat history (list of gr.ChatMessage dicts)

        Yields:
            Tuple of (updated_history, empty_input)
        """
        try:
            # Initialize agent
            self._init_agent()

            # Check if message is empty (can happen on cancellation)
            if not message or not message.strip():
                yield history, ""
                return

            # Note: Video context is already set via Context Catalog selection
            # The selected_video_ids global state is maintained by the UI

            import time
            import threading
            import re

            start_time = time.time()

            # Add user message to history
            history = history + [gr.ChatMessage(role="user", content=message)]
            yield history, ""

            # Build initial status message
            from tools.videodb_query_tool import get_selected_video_ids
            from tools.pdf_rag_tool import get_selected_pdf_files
            selected_videos = get_selected_video_ids()
            selected_pdfs = get_selected_pdf_files()

            context_info = ""
            if selected_videos or selected_pdfs:
                context_info = f"📋 **컨텍스트:** "
                if selected_videos:
                    context_info += f"영상 {len(selected_videos)}개 "
                if selected_pdfs:
                    context_info += f"PDF {len(selected_pdfs)}개"
            else:
                context_info = "⚠️ 선택된 컨텍스트 없음"

            # Add thinking status with pending spinner
            history = history + [gr.ChatMessage(
                role="assistant",
                content=f"🧠 **처리 중...**\n\n{context_info}",
                metadata={"title": "⏳ 에이전트 생각 중...", "status": "pending"}
            )]
            yield history, ""

            # Shared state for streaming updates
            processed_steps = []
            agent_result = [None]
            agent_error = [None]
            execution_complete = [False]

            # Helper functions for formatting step content
            def _clean_model_output(model_output: str) -> str:
                """Clean up model output by removing trailing tags and extra backticks."""
                if not model_output:
                    return ""
                model_output = model_output.strip()
                model_output = re.sub(r"```\s*<end_code>", "```", model_output)
                model_output = re.sub(r"<end_code>\s*```", "```", model_output)
                model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
                return model_output.strip()

            def _format_code_content(content: str) -> str:
                """Format code content as Python code block if not already formatted."""
                content = content.strip()
                content = re.sub(r"```.*?\n", "", content)
                content = re.sub(r"\s*<end_code>\s*", "", content)
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"
                return content

            def _get_step_footnote(step, step_name: str) -> str:
                """Get a footnote string for a step log with duration and token information."""
                footnote = f"**{step_name}**"
                if hasattr(step, 'token_usage') and step.token_usage is not None:
                    footnote += f" | Input: {step.token_usage.input_tokens:,} | Output: {step.token_usage.output_tokens:,}"
                if hasattr(step, 'timing') and step.timing and step.timing.duration:
                    footnote += f" | Duration: {round(float(step.timing.duration), 2)}s"
                return f'<span style="color: #bbbbc2; font-size: 12px;">{footnote}</span>'

            def format_step_as_messages(step):
                """Convert a step to a list of gr.ChatMessage objects with collapsible accordions."""
                messages = []
                try:
                    from smolagents.memory import ActionStep, PlanningStep

                    if isinstance(step, PlanningStep):
                        # Planning step accordion
                        messages.append(gr.ChatMessage(
                            role="assistant",
                            content=step.plan,
                            metadata={"title": "📋 계획 단계", "status": "done"}
                        ))
                        messages.append(gr.ChatMessage(
                            role="assistant",
                            content=_get_step_footnote(step, "계획 단계"),
                            metadata={"status": "done"}
                        ))

                    elif isinstance(step, ActionStep):
                        step_name = f"단계 {step.step_number}"

                        # Step number header
                        messages.append(gr.ChatMessage(
                            role="assistant",
                            content=f"**{step_name}**",
                            metadata={"status": "done"}
                        ))

                        # Model reasoning/thought (if available)
                        if getattr(step, "model_output", ""):
                            model_output = _clean_model_output(step.model_output)
                            if model_output:
                                messages.append(gr.ChatMessage(
                                    role="assistant",
                                    content=model_output,
                                    metadata={"title": "💭 에이전트 추론", "status": "done"}
                                ))

                        # Tool calls with collapsible accordion
                        if getattr(step, "tool_calls", []):
                            for tool_call in step.tool_calls:
                                tool_name = tool_call.name
                                args = tool_call.arguments

                                # Format arguments
                                if isinstance(args, dict):
                                    if tool_name == "python_interpreter":
                                        content = _format_code_content(str(args.get("code", str(args))))
                                    else:
                                        # Format dict arguments nicely
                                        content = "```json\n" + json.dumps(args, indent=2, default=str) + "\n```"
                                else:
                                    content = str(args).strip()

                                messages.append(gr.ChatMessage(
                                    role="assistant",
                                    content=content,
                                    metadata={"title": f"🛠️ 도구 사용: {tool_name}", "status": "done"}
                                ))

                        # Observations/execution logs
                        if getattr(step, "observations", "") and step.observations.strip():
                            log_content = step.observations.strip()
                            log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                            if log_content:
                                # Truncate very long observations
                                display_content = log_content[:2000] + ("..." if len(log_content) > 2000 else "")
                                messages.append(gr.ChatMessage(
                                    role="assistant",
                                    content=f"```\n{display_content}\n```",
                                    metadata={"title": "📝 도구 출력", "status": "done"}
                                ))

                        # Errors
                        if getattr(step, "error", None):
                            messages.append(gr.ChatMessage(
                                role="assistant",
                                content=str(step.error),
                                metadata={"title": "💥 오류", "status": "done"}
                            ))

                        # Step footnote
                        messages.append(gr.ChatMessage(
                            role="assistant",
                            content=_get_step_footnote(step, step_name),
                            metadata={"status": "done"}
                        ))

                        # Separator
                        messages.append(gr.ChatMessage(
                            role="assistant",
                            content="-----",
                            metadata={"status": "done"}
                        ))

                except Exception as e:
                    print(f"Error formatting step: {e}")
                    messages.append(gr.ChatMessage(
                        role="assistant",
                        content=f"단계 처리 오류: {str(e)}",
                        metadata={"status": "done"}
                    ))

                return messages

            # Callback to capture steps
            def step_callback(step):
                processed_steps.append(step)

            # Execute agent in background thread
            def run_agent():
                try:
                    # Register callbacks
                    if hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'step_callbacks'):
                        from smolagents.memory import ActionStep, PlanningStep
                        self.agent.agent.step_callbacks.register(ActionStep, step_callback)
                        self.agent.agent.step_callbacks.register(PlanningStep, step_callback)

                    # Prepare enhanced message with context
                    enhanced_message = self._prepare_message_with_context(message)
                    result = self.agent.run(enhanced_message)
                    agent_result[0] = str(result)
                except Exception as e:
                    agent_error[0] = e
                finally:
                    execution_complete[0] = True

            # Start agent execution
            agent_thread = threading.Thread(target=run_agent, daemon=True)
            agent_thread.start()

            # Stream updates - convert processed steps to ChatMessages
            last_step_count = 0
            base_history_len = len(history) - 1  # Position before "thinking" message

            while not execution_complete[0] or len(processed_steps) > last_step_count:
                if len(processed_steps) > last_step_count:
                    # Build history with all processed steps
                    new_history = history[:base_history_len]  # Keep user message

                    # Add all step messages
                    for step in processed_steps:
                        step_messages = format_step_as_messages(step)
                        new_history.extend(step_messages)

                    # Add "still processing" indicator
                    new_history.append(gr.ChatMessage(
                        role="assistant",
                        content="⏳ 처리 중...",
                        metadata={"title": "🔄 에이전트 작업 중", "status": "pending"}
                    ))

                    history = new_history
                    yield history, ""
                    last_step_count = len(processed_steps)

                time.sleep(0.1)

            agent_thread.join(timeout=1.0)

            # Build final response
            duration = time.time() - start_time

            # Rebuild history with all steps
            final_history = history[:base_history_len]  # Keep user message

            # Add all step messages
            for step in processed_steps:
                step_messages = format_step_as_messages(step)
                final_history.extend(step_messages)

            # Add final answer
            if agent_error[0]:
                error_str = str(agent_error[0])
                from smolagents.agents import AgentError
                is_interruption = isinstance(agent_error[0], AgentError) and "interrupted" in error_str.lower()

                if is_interruption:
                    final_history.append(gr.ChatMessage(
                        role="assistant",
                        content=f"⏹️ 사용자에 의해 {duration:.1f}초 후 에이전트 실행 중지됨",
                        metadata={"status": "done"}
                    ))
                else:
                    final_history.append(gr.ChatMessage(
                        role="assistant",
                        content=f"❌ 오류: {error_str}\n\n자세한 내용은 콘솔을 확인하세요.",
                        metadata={"title": "💥 오류", "status": "done"}
                    ))
            else:
                response = agent_result[0] if agent_result[0] else "에이전트 실행 완료"
                final_history.append(gr.ChatMessage(
                    role="assistant",
                    content=f"**최종 답변:**\n\n{response}",
                    metadata={"status": "done"}
                ))

            # Add completion summary
            final_history.append(gr.ChatMessage(
                role="assistant",
                content=f'<span style="color: #bbbbc2; font-size: 12px;">✅ {duration:.1f}초 만에 {len(processed_steps)}단계로 완료</span>',
                metadata={"status": "done"}
            ))

            yield final_history, ""

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"❌ **시스템 오류:** {str(e)}\n\n다시 시도하거나 콘솔 로그를 확인하세요."
            history = history + [gr.ChatMessage(
                role="assistant",
                content=error_msg,
                metadata={"title": "💥 시스템 오류", "status": "done"}
            )]
            yield history, ""

    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface

        Returns:
            Gradio Blocks interface
        """
        # Register static paths for serving video files
        from pathlib import Path

        static_paths = []

        # Register all collection video directories
        collections_base = Path("./data/local_videodb/collections").absolute()
        if collections_base.exists():
            for collection_dir in collections_base.iterdir():
                if collection_dir.is_dir():
                    videos_dir = collection_dir / "videos"
                    if videos_dir.exists():
                        static_paths.append(videos_dir)
                        print(f"Registered static path for collection '{collection_dir.name}': {videos_dir}")

        # Set all static paths
        if static_paths:
            gr.set_static_paths(paths=static_paths)

        # JavaScript for wargame map state sync (EXACT copy from wargame_ui.py)
        # Uses hidden textbox bridge pattern for reliable JS-to-Python communication
        # This is a FUNCTION (not IIFE) - called by demo.load()
        wargame_state_sync_js = """
        function() {
            // Listen for postMessage from map iframe
            window.addEventListener('message', function(event) {
                if (event.data && (event.data.type === 'wargame_state_change' || event.data.type === 'wargame_state_update')) {
                    var stateJson = event.data.stateJson || JSON.stringify(event.data.state);

                    // Store in global variable for access by buttons
                    window._wargameMapState = stateJson;

                    // Find the hidden textbox by elem_id and update its value
                    // This triggers Gradio's change event to call Python
                    // Try multiple selectors for compatibility across Gradio versions
                    var textbox = document.querySelector('#map_state_holder textarea') ||
                                  document.querySelector('#map_state_holder input') ||
                                  document.getElementById('map_state_holder');

                    if (textbox) {
                        // For textarea/input elements
                        if (textbox.tagName === 'TEXTAREA' || textbox.tagName === 'INPUT') {
                            textbox.value = stateJson;
                        } else {
                            // For container elements, find the input inside
                            var input = textbox.querySelector('textarea') || textbox.querySelector('input');
                            if (input) {
                                input.value = stateJson;
                                textbox = input;
                            }
                        }

                        // Dispatch input event to trigger Gradio's change detection
                        var inputEvent = new Event('input', { bubbles: true });
                        textbox.dispatchEvent(inputEvent);

                        // Also dispatch change event for redundancy
                        var changeEvent = new Event('change', { bubbles: true });
                        textbox.dispatchEvent(changeEvent);
                    } else {
                        console.warn('Could not find map_state_holder textbox');
                    }
                }
            });
            console.log('Wargame message listener initialized - using textbox bridge');
        }
        """

        with gr.Blocks(
            title=self.config.get("ui", {}).get("title", "Battlefield Reconnaissance Video Agent")
        ) as demo:
            # Inject custom CSS and init_js via HTML component
            # Note: wargame_state_sync_js is loaded via demo.load() at the end (like wargame_ui.py)

            gr.Markdown(f"""
# {self.config.get("ui", {}).get("title", "Battlefield Reconnaissance Video Agent")}

{self.config.get("ui", {}).get("description", "드론 영상을 업로드하고 분석합니다")}

## 🚀 빠른 시작
1. **영상 업로드**: 분석할 드론 영상을 업로드합니다
2. **(선택) PDF 추가**: 참고 문서를 컨텍스트로 추가합니다
3. **채팅**: 분석된 영상에 대해 질문합니다
""")

            # Left Sidebar for Collection Management
            with gr.Sidebar(position="left", width=320, open=True):
                gr.Markdown("## 📂 컬렉션")
                gr.Markdown("활성 컬렉션을 선택하세요")

                # Active Collection Selection
                with gr.Accordion("활성 컬렉션", open=True):
                    gr.Markdown("작업할 컬렉션을 선택하세요:")

                    sidebar_active_collection = gr.Dropdown(
                        label="컬렉션 선택",
                        choices=self.collection_manager.get_collection_choices_for_dropdown(),
                        value=self.collection_state.get_active_collections()[0] if self.collection_state.get_active_collections() else None,
                        interactive=True
                    )

                    with gr.Row():
                        sidebar_refresh_btn = gr.Button("🔄", size="sm", scale=0)

                    sidebar_status = gr.Markdown("")

                # Create New Collection Section
                with gr.Accordion("컬렉션 생성", open=False):
                    sidebar_new_name = gr.Textbox(
                        label="이름",
                        placeholder="훈련 임무 2024"
                    )
                    sidebar_new_desc = gr.Textbox(
                        label="설명",
                        placeholder="선택사항 설명",
                        lines=2
                    )
                    sidebar_create_btn = gr.Button("➕ 생성", variant="primary")
                    sidebar_create_status = gr.Markdown("")

                # Delete Collection Section
                with gr.Accordion("컬렉션 삭제", open=False):
                    gr.Markdown("⚠️ **경고**: 이 작업은 되돌릴 수 없습니다!")

                    sidebar_delete_collection = gr.Dropdown(
                        label="삭제할 컬렉션 선택",
                        choices=self.collection_manager.get_collection_choices_for_dropdown(),
                        value=None,
                        interactive=True
                    )

                    sidebar_force_delete = gr.Checkbox(
                        label="강제 삭제 (영상이 있어도 삭제)",
                        value=False
                    )

                    with gr.Row():
                        sidebar_delete_btn = gr.Button("🗑️ 컬렉션 삭제", variant="stop")

                    sidebar_delete_status = gr.Markdown("")

            # Wire up sidebar events
            sidebar_refresh_btn.click(
                fn=self.refresh_collection_dropdown,
                inputs=[],
                outputs=[sidebar_active_collection]
            ).then(
                fn=lambda: gr.Dropdown(choices=self.collection_manager.get_collection_choices_for_dropdown()),
                inputs=[],
                outputs=[sidebar_delete_collection]
            )

            sidebar_create_btn.click(
                fn=self.create_new_collection,
                inputs=[sidebar_new_name, sidebar_new_desc],
                outputs=[sidebar_create_status, sidebar_active_collection]
            ).then(
                fn=lambda: gr.Dropdown(choices=self.collection_manager.get_collection_choices_for_dropdown()),
                inputs=[],
                outputs=[sidebar_delete_collection]
            )

            sidebar_delete_btn.click(
                fn=self.delete_collection_from_sidebar,
                inputs=[sidebar_delete_collection, sidebar_force_delete],
                outputs=[sidebar_delete_status, sidebar_active_collection]
            ).then(
                fn=lambda: gr.Dropdown(choices=self.collection_manager.get_collection_choices_for_dropdown()),
                inputs=[],
                outputs=[sidebar_delete_collection]
            )

            # Also update the delete dropdown when active collection changes
            sidebar_active_collection.change(
                fn=lambda: gr.Dropdown(choices=self.collection_manager.get_collection_choices_for_dropdown()),
                inputs=[],
                outputs=[sidebar_delete_collection]
            )

            # Store reference to video context components for later binding
            # (will be defined in Context Catalog tab)
            video_context_components = {}

            with gr.Tabs():
                # Tab 1: Video Upload
                with gr.Tab("📹 영상 업로드 및 분석"):
                    gr.Markdown("### 드론 영상 업로드")
                    gr.Markdown("영상은 사이드바에서 선택한 첫 번째 활성 컬렉션에 업로드됩니다")

                    with gr.Row():
                        with gr.Column():
                            video_input = gr.Video(label="영상 파일 업로드")
                            segment_duration = gr.Slider(
                                minimum=3,
                                maximum=300,
                                value=3,
                                step=1,
                                label="세그먼트 길이 (초)"
                            )
                            upload_btn = gr.Button("🔍 영상 분석", variant="primary")

                        with gr.Column():
                            upload_status = gr.Markdown("분석을 시작하려면 영상을 업로드하세요")
                            video_id_output = gr.Textbox(label="영상 ID", interactive=False)

                    with gr.Accordion("📊 상세 결과", open=False):
                        analysis_results = gr.Code(label="분석 결과 (JSON)", language="json")

                    # Wire up upload button (gets collection from sidebar selection)
                    upload_btn.click(
                        fn=self.process_video_upload_from_sidebar,
                        inputs=[video_input, segment_duration],
                        outputs=[upload_status, video_id_output, analysis_results]
                    )

                # Tab 2: Add PDFs
                with gr.Tab("📄 참고 문서 추가"):
                    gr.Markdown("### 컨텍스트용 PDF 문서 추가")

                    with gr.Row():
                        pdf_input = gr.File(label="PDF 업로드", file_types=[".pdf"])
                        add_pdf_btn = gr.Button("📚 지식 베이스에 추가", variant="primary")

                    pdf_status = gr.Markdown("지식 베이스에 추가하려면 PDF를 업로드하세요")

                    add_pdf_btn.click(
                        fn=self.add_pdf_to_system,
                        inputs=[pdf_input],
                        outputs=[pdf_status]
                    )

                # Tab 3: Context Catalog
                with gr.Tab("📚 컨텍스트 카탈로그"):
                    gr.Markdown("### 에이전트용 영상 및 PDF 컨텍스트 선택")
                    gr.Markdown("""
에이전트가 질문에 답변할 때 접근할 수 있는 분석된 영상과 PDF 문서를 선택합니다.
여러 영상과 PDF를 선택하여 교차 컨텍스트 분석이 가능합니다.

**참고:** 사이드바를 사용하여 활성 컬렉션을 관리하세요.
""")

                    with gr.Row():
                        # Video contexts column
                        with gr.Column():
                            gr.Markdown("#### 📹 사용 가능한 영상")

                            # Interactive thumbnail grid
                            video_thumbnails = gr.HTML(
                                value="<p style='text-align: center; color: #666;'>'영상 목록 새로고침'을 클릭하여 썸네일을 로드하세요</p>",
                                label="영상 썸네일"
                            )

                            # Hidden field for JavaScript-backend communication
                            # Use visible=True but hide with CSS to ensure DOM element exists
                            video_selection_state = gr.Textbox(
                                elem_id="video-selection-state",
                                visible=True,
                                value="[]",
                                container=False,
                                show_label=False,
                                elem_classes=["video-selection-hidden"]
                            )

                            # Selection summary
                            video_selection_summary = gr.HTML(
                                elem_id="selection-summary",
                                value="<div id='selection-summary'><p style='color: #666;'>ℹ️ 선택된 영상이 없습니다. 썸네일을 클릭하여 선택하세요.</p></div>"
                            )

                            with gr.Row():
                                refresh_videos_btn = gr.Button("🔄 영상 목록 새로고침", size="sm", scale=1)
                                apply_selection_btn = gr.Button("✅ 선택 적용", size="sm", variant="primary", scale=1)

                            video_selection_status = gr.Markdown("사용 가능한 영상을 로드하려면 '새로고침'을 클릭하세요")

                        # PDF contexts column
                        with gr.Column():
                            gr.Markdown("#### 📄 사용 가능한 PDF")
                            pdf_selector = gr.CheckboxGroup(
                                choices=[],
                                label="검색할 PDF 선택",
                                value=[],
                                interactive=True
                            )

                            with gr.Row():
                                refresh_pdfs_btn = gr.Button("🔄 PDF 목록 새로고침", size="sm", scale=1)
                                delete_selected_pdfs_btn = gr.Button("🗑️ 선택된 PDF 삭제", size="sm", variant="stop", scale=1)

                            pdf_selection_status = gr.Markdown("사용 가능한 PDF를 로드하려면 '새로고침'을 클릭하세요")

                    # Hidden fields for delete triggers
                    video_delete_id = gr.Textbox(
                        elem_id="video-delete-id",
                        visible=True,
                        value="",
                        container=False,
                        show_label=False,
                        elem_classes=["video-selection-hidden"]
                    )

                    pdf_delete_filename = gr.Textbox(
                        elem_id="pdf-delete-filename",
                        visible=True,
                        value="",
                        container=False,
                        show_label=False,
                        elem_classes=["video-selection-hidden"]
                    )

                    # Delete status displays
                    delete_status = gr.Markdown("", visible=True)

                    # Context summary
                    gr.Markdown("### 📊 현재 선택")
                    selected_contexts_summary = gr.Markdown("선택된 컨텍스트 없음")

                    # Wire up collection change to auto-refresh video contexts
                    sidebar_active_collection.change(
                        fn=self.update_active_collection_single,
                        inputs=[sidebar_active_collection],
                        outputs=[sidebar_status, video_thumbnails, video_selection_state, video_selection_status]
                    )

                    # Wire up refresh buttons
                    refresh_videos_btn.click(
                        fn=self.refresh_video_contexts,
                        inputs=[],
                        outputs=[video_thumbnails, video_selection_state, video_selection_status]
                    )

                    refresh_pdfs_btn.click(
                        fn=self.refresh_pdf_contexts,
                        inputs=[],
                        outputs=[pdf_selector, pdf_selection_status]
                    )

                    # Wire up delete selected PDFs button
                    delete_selected_pdfs_btn.click(
                        fn=self.delete_selected_pdfs,
                        inputs=[pdf_selector],
                        outputs=[delete_status]
                    ).then(
                        fn=self.refresh_pdf_contexts,
                        inputs=[],
                        outputs=[pdf_selector, pdf_selection_status]
                    )

                    # Wire up manual apply button
                    # First, force JavaScript sync, then parse the state
                    apply_selection_btn.click(
                        fn=None,
                        js="() => { forceApplySelection(); return null; }",
                        inputs=None,
                        outputs=None
                    ).then(
                        fn=self.parse_video_selection,
                        inputs=[video_selection_state],
                        outputs=[selected_contexts_summary]
                    )

                    # Wire up selection updates
                    # Parse video selection from JavaScript state (automatic)
                    video_selection_state.change(
                        fn=self.parse_video_selection,
                        inputs=[video_selection_state],
                        outputs=[selected_contexts_summary]
                    )

                    pdf_selector.change(
                        fn=self.update_selected_pdfs,
                        inputs=[pdf_selector],
                        outputs=[selected_contexts_summary]
                    )

                    # Wire up delete triggers
                    # When video_delete_id changes (JavaScript sets it), trigger deletion
                    video_delete_id.change(
                        fn=self.delete_video_context,
                        inputs=[video_delete_id],
                        outputs=[delete_status]
                    ).then(
                        fn=self.refresh_video_contexts,
                        inputs=[],
                        outputs=[video_thumbnails, video_selection_state, video_selection_status]
                    ).then(
                        fn=self.parse_video_selection,  # Parse updated selection state
                        inputs=[video_selection_state],
                        outputs=[selected_contexts_summary]
                    ).then(
                        fn=lambda: "",  # Clear delete trigger
                        inputs=[],
                        outputs=[video_delete_id]
                    )

                    # When pdf_delete_filename changes (JavaScript sets it), trigger deletion
                    pdf_delete_filename.change(
                        fn=self.delete_pdf_context,
                        inputs=[pdf_delete_filename],
                        outputs=[delete_status]
                    ).then(
                        fn=self.refresh_pdf_contexts,
                        inputs=[],
                        outputs=[pdf_selector, pdf_selection_status]
                    ).then(
                        fn=lambda: "",  # Clear delete trigger
                        inputs=[],
                        outputs=[pdf_delete_filename]
                    )

                # Tab 4: Tracking Videos Gallery
                with gr.Tab("🎬 추적 영상"):
                    gr.Markdown("### 객체 추적 영상 갤러리")
                    gr.Markdown("""
객체 탐지 및 추적 시각화가 포함된 추적 영상을 확인합니다.
각 세그먼트는 바운딩 박스와 추적 ID로 탐지된 객체를 표시합니다.
영상은 원본 영상별로 정리되며, 업로드된 영상당 한 행씩 표시됩니다.
""")

                    # Gallery display
                    tracking_gallery = gr.HTML(
                        value="<p style='text-align: center; color: #666; padding: 40px;'>'추적 영상 로드'를 클릭하여 추적 영상을 표시하세요</p>",
                        label="추적 영상"
                    )

                    # Controls
                    with gr.Row():
                        load_tracking_btn = gr.Button("🔄 추적 영상 로드", variant="primary", size="lg")
                        tracking_status = gr.Markdown("추적 영상을 로드하려면 클릭하세요")

                    # Wire up button
                    load_tracking_btn.click(
                        fn=self.render_tracking_gallery,
                        inputs=[],
                        outputs=[tracking_gallery, tracking_status]
                    )

                # Tab 5: Agent Chat with War Game Map
                with gr.Tab("💬 에이전트 질의"):
                    gr.Markdown("### 전술 지도와 정찰 에이전트")
                    gr.Markdown("""
**지도 조작:**
- **+ 아군 / + 적군 버튼**: 설정 패널에서 부대를 추가한 후 지도를 클릭하여 배치
- **경유지 추가 모드**: 먼저 지도에서 부대를 선택한 후 클릭하여 경유지 추가
- **부대 이동**: 부대를 클릭하고 드래그하여 위치 변경
- **부대 아이콘 클릭**: 부대 삭제 (지도에서 생성된 부대만)
- **스크롤 휠**: 확대/축소

**오른쪽 패널**: AI 에이전트로 정찰 데이터를 질의합니다.
""")

                    # Hidden textbox for map state sync (bridge between JS and Python)
                    # The postMessage listener updates this textbox, triggering change events
                    # Note: Use visible="hidden" instead of visible=False for Gradio compatibility
                    # visible=False doesn't render in DOM, visible="hidden" keeps it in DOM but hidden
                    map_state_holder = gr.Textbox(visible="hidden", elem_id="map_state_holder")

                    with gr.Row():
                        # Left column: Map display and Settings Panel below
                        with gr.Column(scale=1):
                            gr.Markdown("#### 🗺️ 전술 지도")
                            wargame_map_display = gr.HTML(
                                value=self.wargame_ui._create_data_uri_iframe(self.wargame_ui._get_map_html(), height=550),
                                elem_id="wargame_map_display"
                            )

                            # War Game Status with Settings Panel (from wargame_ui.py)
                            with gr.Accordion("워게임 상태 및 설정", open=True):
                                gr.Markdown(f"*지도 상태 변경 시 자동 업데이트됩니다. 저장 위치: `{self.wargame_ui._auto_save_path}`*")

                                with gr.Row():
                                    with gr.Column(scale=1):
                                        units_summary_display = gr.Textbox(
                                            label="부대 요약",
                                            value=self.wargame_ui.get_full_status(""),
                                            lines=8,
                                            interactive=False,
                                            elem_id="units_summary_display"
                                        )
                                    with gr.Column(scale=1):
                                        state_json_display = gr.Code(
                                            label="자동 저장된 JSON 상태",
                                            language="json",
                                            value=self.wargame_ui.auto_save_state(""),
                                            lines=8
                                        )

                                map_status = gr.Textbox(label="상태", interactive=False, lines=1)

                                # Settings Panel (moved from left column)
                                with gr.Accordion("설정 패널", open=False):
                                    with gr.Row():
                                        # Left settings: Add Units
                                        with gr.Column(scale=1):
                                            with gr.Accordion("아군 부대 추가", open=True):
                                                friendly_name_input = gr.Textbox(
                                                    label="부대명",
                                                    value="알파 소대",
                                                    placeholder="예: 알파 소대"
                                                )
                                                with gr.Row():
                                                    friendly_type_dropdown = gr.Dropdown(
                                                        label="부대 유형",
                                                        choices=[e.name for e in UnitType],
                                                        value="INFANTRY"
                                                    )
                                                    friendly_echelon_dropdown = gr.Dropdown(
                                                        label="제대",
                                                        choices=[e.name for e in Echelon if e.name != "BATTERY"],
                                                        value="PLATOON"
                                                    )
                                                with gr.Row():
                                                    friendly_lat = gr.Number(label="위도", value=37.57)
                                                    friendly_lon = gr.Number(label="경도", value=126.98)
                                                with gr.Row():
                                                    friendly_designation = gr.Textbox(
                                                        label="부대 표시",
                                                        placeholder="예: 1-1-A"
                                                    )
                                                    friendly_strength = gr.Number(label="병력", value=30)
                                                add_friendly_btn = gr.Button("아군 부대 추가", variant="primary")

                                            with gr.Accordion("적군 부대 추가", open=False):
                                                hostile_name_input = gr.Textbox(
                                                    label="부대명",
                                                    value="적 1세력",
                                                    placeholder="예: 적 전차중대"
                                                )
                                                with gr.Row():
                                                    hostile_type_dropdown = gr.Dropdown(
                                                        label="부대 유형",
                                                        choices=[e.name for e in UnitType],
                                                        value="ARMOR"
                                                    )
                                                    hostile_echelon_dropdown = gr.Dropdown(
                                                        label="제대",
                                                        choices=[e.name for e in Echelon if e.name != "BATTERY"],
                                                        value="COMPANY"
                                                    )
                                                with gr.Row():
                                                    hostile_lat = gr.Number(label="위도", value=37.55)
                                                    hostile_lon = gr.Number(label="경도", value=126.99)
                                                with gr.Row():
                                                    hostile_designation = gr.Textbox(
                                                        label="부대 표시",
                                                        placeholder="예: RED-1"
                                                    )
                                                    hostile_strength = gr.Number(label="병력", value=0)
                                                add_hostile_btn = gr.Button("적군 부대 추가", variant="stop")

                                        # Right settings: Map, Scenario, Actions
                                        with gr.Column(scale=1):
                                            with gr.Accordion("지도 설정", open=False):
                                                with gr.Row():
                                                    map_lat_input = gr.Number(label="위도", value=37.5665)
                                                    map_lon_input = gr.Number(label="경도", value=126.9780)
                                                map_zoom_input = gr.Slider(
                                                    label="확대 수준",
                                                    minimum=1,
                                                    maximum=18,
                                                    value=10,
                                                    step=1
                                                )
                                                create_map_btn = gr.Button("지도 생성/초기화", variant="primary")

                                            with gr.Accordion("시나리오 관리", open=False):
                                                scenario_file = gr.File(
                                                    label="시나리오 불러오기",
                                                    file_types=[".json"]
                                                )
                                                load_scenario_btn = gr.Button("시나리오 불러오기")
                                                save_scenario_name = gr.Textbox(
                                                    label="시나리오 저장",
                                                    placeholder="scenario.json"
                                                )
                                                save_scenario_btn = gr.Button("시나리오 저장")

                                            with gr.Accordion("작업", open=True):
                                                sync_map_btn = gr.Button("지도에서 동기화", variant="secondary")
                                                gr.Markdown("*동기화하면 지도에서 클릭한 부대가 시나리오에 저장됩니다*", elem_classes=["hint-text"])
                                                with gr.Row():
                                                    clear_units_btn = gr.Button("부대 지우기 (설정)")
                                                    clear_all_map_btn = gr.Button("모두 지우기", variant="secondary")
                                                refresh_summary_btn = gr.Button("상태 새로고침")

                        # Right column: Agent Chat
                        with gr.Column(scale=1):
                            gr.Markdown("#### 💬 정찰 에이전트")

                            # Note: In Gradio 6.0+, "messages" format is the default (tuples format removed)
                            # ChatMessage with metadata automatically enables collapsible accordions
                            chatbot = gr.Chatbot(
                                label="에이전트 채팅",
                                height=400
                            )

                            with gr.Row():
                                msg_input = gr.Textbox(
                                    label="질문",
                                    placeholder="영상이나 전술 상황에 대해 질문하세요...",
                                    scale=4
                                )
                                with gr.Column(scale=0, min_width=80):
                                    send_btn = gr.Button("전송", variant="primary")
                                    stop_btn = gr.Button("중지", variant="stop")

                            stop_status = gr.Markdown("", visible=False)

                            gr.Markdown("#### 💡 예시")
                            examples = self.config.get("ui", {}).get("examples", [
                                "전차가 몇 대 나타나나요?",
                                "병사의 움직임을 설명해주세요",
                                "트럭이 있는 세그먼트를 찾아주세요"
                            ])

                            with gr.Row():
                                for example in examples[:2]:
                                    gr.Button(example, size="sm").click(
                                        fn=lambda x: x,
                                        inputs=gr.State(example),
                                        outputs=msg_input
                                    )

                            gr.Markdown("*접을 수 있는 섹션을 클릭하여 에이전트 추론 과정을 확인하세요.*", elem_classes=["hint-text"])

                    # =====================================================
                    # Map Event Handlers (from wargame_ui.py pattern)
                    # =====================================================

                    # Helper function to update full status and auto-save JSON
                    def update_full_status():
                        full_status = self.wargame_ui.get_full_status("")
                        json_state = self.wargame_ui.auto_save_state("")
                        return full_status, json_state

                    # Helper to update status with map state from JavaScript
                    def update_status_with_map_state(map_state_json: str):
                        full_status = self.wargame_ui.get_full_status(map_state_json)
                        json_state = self.wargame_ui.auto_save_state(map_state_json)
                        return full_status, json_state

                    # Helper function to update all status displays and auto-save JSON
                    def update_all_status(map_state_json: str):
                        full_status = self.wargame_ui.get_full_status(map_state_json)
                        brief_status = self.wargame_ui.get_brief_status(map_state_json)
                        json_state = self.wargame_ui.auto_save_state(map_state_json)
                        return full_status, brief_status, json_state

                    # =====================================================
                    # Periodic State Sync using Timer (reliable backup)
                    # This fetches state from JS every 2 seconds
                    # =====================================================
                    state_sync_timer = gr.Timer(value=2.0, active=True)

                    def periodic_sync_handler(current_state: str):
                        """Called by timer - updates status displays with JS state"""
                        if current_state and current_state != "{}":
                            try:
                                full_status = self.wargame_ui.get_full_status(current_state)
                                json_state = self.wargame_ui.auto_save_state(current_state)
                                brief_status = self.wargame_ui.get_brief_status(current_state)
                                return full_status, brief_status, json_state
                            except Exception as e:
                                print(f"[WargameSync] Error in periodic sync: {e}")
                        return gr.update(), gr.update(), gr.update()

                    # Timer tick fetches state from JS global variable
                    state_sync_timer.tick(
                        fn=None,
                        inputs=None,
                        outputs=[map_state_holder],
                        js="""
                        () => {
                            if (window._wargameMapState && window._wargameMapState !== '{}') {
                                return window._wargameMapState;
                            }
                            return '{}';
                        }
                        """
                    ).then(
                        fn=periodic_sync_handler,
                        inputs=[map_state_holder],
                        outputs=[units_summary_display, map_status, state_json_display]
                    )

                    # Create/Reset Map
                    create_map_btn.click(
                        fn=self.wargame_ui.create_map,
                        inputs=[map_lat_input, map_lon_input, map_zoom_input],
                        outputs=[wargame_map_display]
                    ).then(
                        fn=update_full_status,
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Add Friendly Unit
                    def add_friendly_unit_handler(name, unit_type, echelon, lat, lon, designation, strength):
                        map_html, status = self.wargame_ui.add_friendly_unit(
                            name=name,
                            unit_type=unit_type,
                            echelon=echelon,
                            lat=lat,
                            lon=lon,
                            designation=designation,
                            strength=strength
                        )
                        return map_html, status

                    add_friendly_btn.click(
                        fn=add_friendly_unit_handler,
                        inputs=[friendly_name_input, friendly_type_dropdown, friendly_echelon_dropdown,
                                friendly_lat, friendly_lon, friendly_designation, friendly_strength],
                        outputs=[wargame_map_display, map_status]
                    ).then(
                        fn=update_full_status,
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Add Hostile Unit
                    def add_hostile_unit_handler(name, unit_type, echelon, lat, lon, designation, strength):
                        map_html, status = self.wargame_ui.add_hostile_unit(
                            name=name,
                            unit_type=unit_type,
                            echelon=echelon,
                            lat=lat,
                            lon=lon,
                            designation=designation,
                            strength=strength
                        )
                        return map_html, status

                    add_hostile_btn.click(
                        fn=add_hostile_unit_handler,
                        inputs=[hostile_name_input, hostile_type_dropdown, hostile_echelon_dropdown,
                                hostile_lat, hostile_lon, hostile_designation, hostile_strength],
                        outputs=[wargame_map_display, map_status]
                    ).then(
                        fn=update_full_status,
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Clear Units (Settings Panel only)
                    clear_units_btn.click(
                        fn=self.wargame_ui.clear_all_units,
                        outputs=[wargame_map_display, map_status]
                    ).then(
                        fn=update_full_status,
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Clear All (entire map)
                    clear_all_map_btn.click(
                        fn=self.wargame_ui.clear_map,
                        outputs=[wargame_map_display, map_status]
                    ).then(
                        fn=update_full_status,
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Load Scenario
                    load_scenario_btn.click(
                        fn=self.wargame_ui.load_scenario,
                        inputs=[scenario_file],
                        outputs=[wargame_map_display, map_status]
                    ).then(
                        fn=update_full_status,
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Save Scenario
                    save_scenario_btn.click(
                        fn=self.wargame_ui.save_scenario,
                        inputs=[save_scenario_name],
                        outputs=[map_status]
                    )

                    # Refresh Status: get current map state from stored variable
                    refresh_summary_btn.click(
                        fn=None,
                        inputs=None,
                        outputs=[map_state_holder],
                        js="""
                        () => {
                            if (window._wargameMapState) {
                                return window._wargameMapState;
                            }
                            return '{}';
                        }
                        """
                    ).then(
                        fn=update_status_with_map_state,
                        inputs=[map_state_holder],
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Sync button: get state from stored variable and sync to Python
                    sync_map_btn.click(
                        fn=None,
                        inputs=None,
                        outputs=[map_state_holder],
                        js="""
                        () => {
                            if (window._wargameMapState) {
                                return window._wargameMapState;
                            }
                            return '{}';
                        }
                        """
                    ).then(
                        fn=self.wargame_ui.sync_from_map,
                        inputs=[map_state_holder],
                        outputs=[wargame_map_display, map_status]
                    ).then(
                        fn=update_status_with_map_state,
                        inputs=[map_state_holder],
                        outputs=[units_summary_display, state_json_display]
                    )

                    # Event-driven status update: when JS updates the hidden textbox,
                    # Gradio's change event triggers this Python callback
                    # This auto-saves to JSON and updates all status displays
                    map_state_holder.change(
                        fn=update_all_status,
                        inputs=[map_state_holder],
                        outputs=[units_summary_display, map_status, state_json_display]
                    )

                    # Chat interactions with streaming (inline thinking display)
                    msg_input.submit(
                        fn=self.chat_with_agent,
                        inputs=[msg_input, chatbot],
                        outputs=[chatbot, msg_input]
                    )

                    send_btn.click(
                        fn=self.chat_with_agent,
                        inputs=[msg_input, chatbot],
                        outputs=[chatbot, msg_input]
                    )

                    # Wire up stop button to interrupt agent execution
                    stop_btn.click(
                        fn=self.stop_agent,
                        inputs=[],
                        outputs=[stop_status]
                    )

            gr.Markdown("""
---
### ℹ️ 시스템 정보
- **객체 탐지 및 추적**: SAM3 (병사, 전차, 트럭)
- **임베딩**: PE-Core-L14-336 (1024차원)
- **이벤트 설명**: SmolVLM2 2.2B Instruct
- **에이전트**: EXAONE-4.0-32B-AWQ with CodeAgent
""")

            # Load JavaScript for message listener on page load (same as wargame_ui.py)
            # The JS listens for postMessage from iframe and updates the hidden textbox
            demo.load(fn=None, js=wargame_state_sync_js)

        return demo

    def launch(self, preload_models: bool = True, preload_agent: bool = True, **kwargs):
        """
        Launch Gradio interface

        Args:
            preload_models: Whether to pre-load all models before launching
            preload_agent: Whether to pre-load agent model (optional, as it's large)
            **kwargs: Additional arguments for demo.launch()
        """
        # JavaScript for interactive thumbnail selection (loaded via demo.load)
        init_js = """
        // Global state for video selection
        let selectedVideos = new Set();

        function initializeSelection(selectedIdsJson) {
            // Initialize selectedVideos Set from JSON string
            try {
                console.log('[VideoSelector] Initializing selection with:', selectedIdsJson);
                const selectedIds = JSON.parse(selectedIdsJson);
                selectedVideos = new Set(selectedIds);
                console.log('[VideoSelector] Initialized selectedVideos:', Array.from(selectedVideos));
                updateAllCardStates();
                updateSelectionSummary();
            } catch (e) {
                console.error('[VideoSelector] Error initializing selection:', e);
                selectedVideos = new Set();
            }
        }

        function findHiddenField() {
            // Strategy 1: Try elem_id directly
            const elemById = document.getElementById('video-selection-state');
            if (elemById) {
                console.log('[VideoSelector] Found element by ID:', elemById.tagName);

                // If it's the container, find textarea inside
                if (elemById.tagName === 'LABEL' || elemById.tagName === 'DIV') {
                    const textarea = elemById.querySelector('textarea');
                    const input = elemById.querySelector('input[type="text"]');
                    if (textarea) {
                        console.log('[VideoSelector] Found textarea inside container');
                        return textarea;
                    }
                    if (input) {
                        console.log('[VideoSelector] Found input inside container');
                        return input;
                    }
                }

                // If it's already the input element
                if (elemById.tagName === 'TEXTAREA' || elemById.tagName === 'INPUT') {
                    console.log('[VideoSelector] Element is already textarea/input');
                    return elemById;
                }
            }

            // Strategy 2: Try by class name
            const elemByClass = document.querySelector('.video-selection-hidden');
            if (elemByClass) {
                console.log('[VideoSelector] Found element by class');
                if (elemByClass.tagName === 'TEXTAREA' || elemByClass.tagName === 'INPUT') {
                    return elemByClass;
                }
                const textarea = elemByClass.querySelector('textarea');
                const input = elemByClass.querySelector('input[type="text"]');
                if (textarea) return textarea;
                if (input) return input;
            }

            // Strategy 3: Try multiple specific selectors
            const selectors = [
                '#video-selection-state textarea',
                '#video-selection-state input',
                '.video-selection-hidden textarea',
                '.video-selection-hidden input',
                'textarea[elem_id="video-selection-state"]',
                'input[elem_id="video-selection-state"]',
                'label[id="video-selection-state"] textarea',
                'label[id="video-selection-state"] input'
            ];

            for (const selector of selectors) {
                try {
                    const elem = document.querySelector(selector);
                    if (elem) {
                        console.log('[VideoSelector] Found hidden field with selector:', selector);
                        return elem;
                    }
                } catch (e) {
                    // Invalid selector, continue
                }
            }

            // Strategy 4: Search all textareas/inputs for one with empty/json value
            console.warn('[VideoSelector] Trying fallback search through all form elements...');
            const allTextareas = document.querySelectorAll('textarea');
            for (const textarea of allTextareas) {
                const val = textarea.value.trim();
                if (val === '[]' || (val.startsWith('[') && val.endsWith(']'))) {
                    // Check if it's near our video thumbnails section
                    const parent = textarea.closest('[id*="context"]') || textarea.closest('.video-grid');
                    if (parent || textarea.style.display === 'none' || textarea.offsetHeight === 0) {
                        console.log('[VideoSelector] Found likely candidate textarea by content pattern');
                        return textarea;
                    }
                }
            }

            console.error('[VideoSelector] Could not find hidden field with any strategy');
            console.log('[VideoSelector] Available elements with video-selection:', document.querySelectorAll('[id*="video-selection"]'));
            return null;
        }

        function syncSelectionToBackend(retryCount = 0) {
            const hiddenField = findHiddenField();

            if (!hiddenField) {
                if (retryCount < 5) {
                    // Increase max retries to 5 and use exponential backoff
                    const delay = 300 * Math.pow(1.5, retryCount); // 300ms, 450ms, 675ms, 1012ms, 1518ms
                    console.log('[VideoSelector] Retrying to find hidden field... attempt', retryCount + 1, 'waiting', Math.round(delay), 'ms');
                    setTimeout(() => syncSelectionToBackend(retryCount + 1), delay);
                    return;
                }
                console.error('[VideoSelector] Failed to sync selection after 5 attempts');
                console.error('[VideoSelector] Possible solutions:');
                console.error('  1. Click the "✅ Apply Selection" button manually');
                console.error('  2. Run debugVideoSelector() in console to diagnose');
                console.error('  3. Refresh the page and try again');
                return;
            }

            const selectionJson = JSON.stringify(Array.from(selectedVideos));
            console.log('[VideoSelector] Syncing selection to backend:', selectionJson);
            console.log('[VideoSelector] Target field:', hiddenField.tagName, 'with value:', hiddenField.value);

            // Update the value
            hiddenField.value = selectionJson;

            // Dispatch multiple event types to ensure Gradio detects the change
            // Using both Event and InputEvent for maximum compatibility
            const eventTypes = ['input', 'change'];
            eventTypes.forEach(eventType => {
                // Try standard Event
                const event1 = new Event(eventType, { bubbles: true, cancelable: true });
                hiddenField.dispatchEvent(event1);

                // Also try InputEvent for 'input' type
                if (eventType === 'input' && typeof InputEvent !== 'undefined') {
                    const event2 = new InputEvent('input', { bubbles: true, cancelable: true });
                    hiddenField.dispatchEvent(event2);
                }
            });

            // Force focus and blur to trigger change detection
            try {
                hiddenField.focus();
                hiddenField.blur();
            } catch (e) {
                // Element might not be focusable, that's okay
            }

            console.log('[VideoSelector] Selection synced successfully');
        }

        function toggleVideoSelection(videoId) {
            console.log('[VideoSelector] Toggle selection for video:', videoId);

            const card = document.querySelector(`[data-video-id="${videoId}"]`);
            if (!card) {
                console.error('[VideoSelector] Card not found for video:', videoId);
                return;
            }

            const indicator = card.querySelector('.selection-indicator');

            if (selectedVideos.has(videoId)) {
                // Deselect
                selectedVideos.delete(videoId);
                card.classList.remove('selected');
                if (indicator) indicator.textContent = '';
                console.log('[VideoSelector] Deselected:', videoId);
            } else {
                // Select
                selectedVideos.add(videoId);
                card.classList.add('selected');
                if (indicator) indicator.textContent = '✓';
                console.log('[VideoSelector] Selected:', videoId);
            }

            console.log('[VideoSelector] Current selection:', Array.from(selectedVideos));

            // Update summary display
            updateSelectionSummary();

            // Sync to backend with retry logic
            syncSelectionToBackend();
        }

        function updateAllCardStates() {
            // Update all cards to match selectedVideos Set
            console.log('[VideoSelector] Updating all card states');
            document.querySelectorAll('.video-card').forEach(card => {
                const videoId = card.getAttribute('data-video-id');
                const indicator = card.querySelector('.selection-indicator');

                if (selectedVideos.has(videoId)) {
                    card.classList.add('selected');
                    if (indicator) indicator.textContent = '✓';
                } else {
                    card.classList.remove('selected');
                    if (indicator) indicator.textContent = '';
                }
            });
        }

        function updateSelectionSummary() {
            const count = selectedVideos.size;
            const summaryEl = document.querySelector('#selection-summary');
            if (summaryEl) {
                if (count === 0) {
                    summaryEl.innerHTML = '<p style="color: #666;">ℹ️ No videos selected. Click on thumbnails to select.</p>';
                } else {
                    const videoList = Array.from(selectedVideos).map(id => id.substring(0, 12) + '...').join(', ');
                    summaryEl.innerHTML = `<p style="color: #4CAF50; font-weight: bold;">✅ ${count} video(s) selected: ${videoList}</p>`;
                }
            }
        }

        function getSelectedVideosJson() {
            // Function for manual Apply button - returns current selection as JSON
            console.log('[VideoSelector] Manual apply - current selection:', Array.from(selectedVideos));
            return JSON.stringify(Array.from(selectedVideos));
        }

        function forceApplySelection() {
            // Force synchronization with retry
            console.log('[VideoSelector] Forcing selection sync via Apply button');
            syncSelectionToBackend(0);
        }

        function findGradioTextbox(elemId) {
            // Helper to find Gradio textbox input by elem_id
            // Gradio wraps inputs in containers, so we need to search carefully
            const strategies = [
                // Strategy 1: Direct ID on textarea/input
                () => document.querySelector(`textarea#${elemId}, input#${elemId}`),
                // Strategy 2: Container with ID containing textarea/input
                () => {
                    const container = document.getElementById(elemId);
                    if (container) {
                        return container.querySelector('textarea') || container.querySelector('input[type="text"]');
                    }
                    return null;
                },
                // Strategy 3: Label with ID containing textarea/input
                () => {
                    const label = document.querySelector(`label#${elemId}`);
                    if (label) {
                        return label.querySelector('textarea') || label.querySelector('input[type="text"]');
                    }
                    return null;
                },
                // Strategy 4: Div with data-testid or similar
                () => document.querySelector(`[data-testid="${elemId}"] textarea, [data-testid="${elemId}"] input`),
                // Strategy 5: Any element with matching ID containing input
                () => {
                    const elem = document.querySelector(`[id="${elemId}"]`);
                    if (elem) {
                        if (elem.tagName === 'TEXTAREA' || elem.tagName === 'INPUT') return elem;
                        return elem.querySelector('textarea') || elem.querySelector('input[type="text"]');
                    }
                    return null;
                }
            ];

            for (const strategy of strategies) {
                const result = strategy();
                if (result) {
                    console.log(`[findGradioTextbox] Found ${elemId} using strategy`);
                    return result;
                }
            }

            console.error(`[findGradioTextbox] Could not find element: ${elemId}`);
            return null;
        }

        function triggerGradioChange(element, value) {
            // Set value and trigger Gradio change detection
            if (!element) return false;

            element.value = value;

            // Dispatch multiple event types for Gradio compatibility
            ['input', 'change'].forEach(eventType => {
                element.dispatchEvent(new Event(eventType, { bubbles: true, cancelable: true }));
            });

            // Also try InputEvent for better compatibility
            if (typeof InputEvent !== 'undefined') {
                element.dispatchEvent(new InputEvent('input', { bubbles: true, cancelable: true, data: value }));
            }

            return true;
        }

        function deleteVideoContext(videoId, videoName) {
            // Confirm deletion
            const confirmMsg = `Are you sure you want to delete the video "${videoName}"?\n\nVideo ID: ${videoId}\n\nThis will permanently remove:\n- Original video file\n- All segments and metadata\n- Tracking videos\n- FAISS embeddings\n\nThis action cannot be undone!`;

            if (!confirm(confirmMsg)) {
                console.log('[VideoDelete] Video deletion cancelled by user');
                return;
            }

            console.log('[VideoDelete] Deleting video:', videoId);

            // Find and update the hidden field
            const deleteField = findGradioTextbox('video-delete-id');
            if (deleteField) {
                if (triggerGradioChange(deleteField, videoId)) {
                    console.log('[VideoDelete] Video deletion triggered for:', videoId);
                } else {
                    console.error('[VideoDelete] Failed to trigger change event');
                    alert('Error: Could not trigger deletion. Please refresh the page and try again.');
                }
            } else {
                console.error('[VideoDelete] Could not find delete trigger field');
                // Try to list what we can find for debugging
                console.log('[VideoDelete] Available elements:', {
                    byId: document.getElementById('video-delete-id'),
                    textareas: document.querySelectorAll('textarea').length,
                    inputs: document.querySelectorAll('input[type="text"]').length
                });
                alert('Error: Could not find delete field. Please refresh the page and try again.');
            }
        }

        function deletePdfContext(pdfFilename) {
            // Confirm deletion
            const confirmMsg = `Are you sure you want to delete the PDF "${pdfFilename}"?\n\nThis will permanently remove:\n- PDF file\n- All indexed chunks\n- Metadata\n\nThis action cannot be undone!`;

            if (!confirm(confirmMsg)) {
                console.log('[PDFDelete] PDF deletion cancelled by user');
                return;
            }

            console.log('[PDFDelete] Deleting PDF:', pdfFilename);

            // Find and update the hidden field
            const deleteField = findGradioTextbox('pdf-delete-filename');
            if (deleteField) {
                if (triggerGradioChange(deleteField, pdfFilename)) {
                    console.log('[PDFDelete] PDF deletion triggered for:', pdfFilename);
                } else {
                    console.error('[PDFDelete] Failed to trigger change event');
                    alert('Error: Could not trigger deletion. Please refresh the page and try again.');
                }
            } else {
                console.error('[PDFDelete] Could not find delete trigger field');
                alert('Error: Could not find delete field. Please refresh the page and try again.');
            }
        }

        // Debug helper - call from console with: debugVideoSelector()
        window.debugVideoSelector = function() {
            console.log('=== VIDEO SELECTOR DEBUG INFO ===');
            console.log('Selected Videos:', Array.from(selectedVideos));
            console.log('Element by ID:', document.getElementById('video-selection-state'));
            console.log('Element by class:', document.querySelector('.video-selection-hidden'));
            console.log('All textareas:', document.querySelectorAll('textarea'));
            console.log('Elements with video-selection in ID:', document.querySelectorAll('[id*="video-selection"]'));

            const field = findHiddenField();
            if (field) {
                console.log('Found field:', field);
                console.log('Field value:', field.value);
                console.log('Field tag:', field.tagName);
                console.log('Field parent:', field.parentElement);
            } else {
                console.log('Field NOT found');
            }
        };

        // Wait for DOM to be fully loaded before allowing interactions
        let domReady = false;
        document.addEventListener('DOMContentLoaded', function() {
            domReady = true;
            console.log('[VideoSelector] DOM ready');
        });

        // If DOMContentLoaded already fired
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
            domReady = true;
        }
        """

        # Custom CSS to fix text visibility issues
        custom_css = """
        /* Hide video selection state textbox - use positioning instead of display:none
           to ensure events still propagate correctly */
        .video-selection-hidden {
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            width: 1px !important;
            height: 1px !important;
            opacity: 0 !important;
            pointer-events: none !important;
            overflow: hidden !important;
        }

        /* Also hide by elem_id as fallback - use same pattern */
        #video-selection-state,
        #video-delete-id,
        #pdf-delete-filename {
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            width: 1px !important;
            height: 1px !important;
            opacity: 0 !important;
            overflow: hidden !important;
        }

        /* Hide map state holder textbox (for JS-to-Python bridge) */
        #map_state_holder {
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            width: 1px !important;
            height: 1px !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        /* Override Gradio theme colors for tracking gallery */
        .tracking-gallery .video-row-title {
            color: #1a1a1a !important;
        }
        .tracking-gallery .video-row-info {
            color: #2d2d2d !important;
        }
        .tracking-gallery .segment-title {
            color: #1a1a1a !important;
        }
        .tracking-gallery .segment-meta {
            color: #2d2d2d !important;
        }
        .tracking-gallery .segment-meta-item {
            color: #2d2d2d !important;
        }
        .tracking-gallery .event-desc {
            color: #1a1a1a !important;
            background: #f8f9ff !important;
        }
        /* Ensure all text in gallery is visible */
        .tracking-gallery * {
            color: inherit;
        }
        """
        
        # Pre-load models if requested
        if preload_models:
            print("Pre-loading models before launching UI...")
            self.model_manager.load_all_models(load_agent=preload_agent)

        # Set up static paths for wargame assets (milsymbol.js, etc.)
        _setup_static_paths()

        demo = self.create_interface()

        # Get launch config from settings
        from pathlib import Path

        # Build allowed paths list for all collections
        allowed_paths = []

        # Add map_mcp assets directory for milsymbol.js
        map_mcp_assets = map_mcp_path / "map_mcp" / "assets"
        if map_mcp_assets.exists():
            allowed_paths.append(str(map_mcp_assets))

        # Add wargame temp directory for HTML files
        wargame_temp = _get_temp_dir()
        allowed_paths.append(str(wargame_temp))

        # Add all collection video directories
        collections_base = Path("./data/local_videodb/collections").absolute()
        if collections_base.exists():
            for collection_dir in collections_base.iterdir():
                if collection_dir.is_dir():
                    videos_dir = collection_dir / "videos"
                    if videos_dir.exists():
                        allowed_paths.append(str(videos_dir))

        launch_config = {
            "server_name": self.config.get("ui", {}).get("server_name", "0.0.0.0"),
            "server_port": self.config.get("ui", {}).get("server_port", 7860),
            "share": self.config.get("ui", {}).get("share", False),
            "allowed_paths": allowed_paths,  # Allow serving tracking videos
            "head": f"<script>{init_js}</script>",
            "css": custom_css,
        }

        # Override with provided kwargs
        launch_config.update(kwargs)

        print(f"Allowed paths for file serving:")
        # Print wargame paths
        if map_mcp_assets.exists():
            print(f"  - Wargame assets: {map_mcp_assets}")
        print(f"  - Wargame temp: {wargame_temp}")
        # Print all video collection paths
        video_paths = [p for p in allowed_paths if 'videos' in str(p)]
        if video_paths:
            print(f"  - Video collections ({len(video_paths)}):")
            for vpath in video_paths:
                collection_name = Path(vpath).parent.name
                print(f"    • {collection_name}: {vpath}")

        print(f"\n{'='*60}")
        print("Launching Battlefield Reconnaissance Video Agent UI")
        print(f"{'='*60}\n")

        demo.launch(**launch_config)


def create_ui(config_path: Optional[str] = None) -> BattlefieldVideoAgentUI:
    """
    Create UI instance

    Args:
        config_path: Path to configuration directory

    Returns:
        BattlefieldVideoAgentUI instance
    """
    return BattlefieldVideoAgentUI(config_path=config_path)


# Main entry point
if __name__ == "__main__":
    ui = create_ui()
    ui.launch()
