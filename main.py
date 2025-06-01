import streamlit as st
import whisper
import pyaudio
import wave
import os
import time
import threading
import tempfile
import shutil
import language_tool_python
from gtts import gTTS
import pygame
import re
import uuid
from pathlib import Path

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Create temp directory for audio files
TEMP_DIR = Path(tempfile.gettempdir()) / "tweakspeare"
TEMP_DIR.mkdir(exist_ok=True)

# Load Whisper model with specific settings for accuracy
@st.cache_resource
def load_model():
    try:
        model = whisper.load_model("base")
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        st.error("Please ensure you have whisper installed: pip install openai-whisper")
        return None

# Load LanguageTool
@st.cache_resource
def load_language_tool():
    try:
        return language_tool_python.LanguageTool('en-US')
    except Exception as e:
        st.error(f"Failed to load LanguageTool: {str(e)}")
        return None

# Check dependencies
def check_dependencies():
    missing_deps = []
    for module, name in [
        ('whisper', 'openai-whisper'),
        ('pyaudio', 'pyaudio'),
        ('language_tool_python', 'language-tool-python'),
        ('gtts', 'gTTS'),
        ('pygame', 'pygame')
    ]:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(name)
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}")
        st.info("Install with: pip install " + " ".join(missing_deps))
        return False
    return True

# List microphones
@st.cache_data
def list_microphones():
    try:
        p = pyaudio.PyAudio()
        microphones = []
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    name = device_info.get('name', f'Device {i}')
                    microphones.append((i, name))
            except:
                continue
        p.terminate()
        return microphones
    except Exception as e:
        st.error(f"Error accessing audio devices: {str(e)}")
        return []

class AudioRecorder:
    def __init__(self, device_index=None):
        self.device_index = device_index
        self.is_recording = False
        self.audio_frames = []
        self.recording_thread = None
        self.temp_file = None
        self.error_message = None
    
    def start_recording(self):
        if self.is_recording:
            return False
        try:
            self.is_recording = True
            self.audio_frames = []
            self.error_message = None
            self.recording_thread = threading.Thread(target=self._record_audio_thread)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            return True
        except Exception as e:
            self.error_message = f"Failed to start recording: {str(e)}"
            self.is_recording = False
            return False
    
    def stop_recording(self):
        if not self.is_recording:
            return None
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0)
        if self.error_message:
            return None, self.error_message
        return self._save_audio_file(), None
    
    def _record_audio_thread(self):
        audio_interface = None
        stream = None
        try:
            audio_interface = pyaudio.PyAudio()
            try:
                device_info = audio_interface.get_device_info_by_index(self.device_index)
                if device_info.get('maxInputChannels', 0) == 0:
                    self.error_message = f"Device {self.device_index} has no input channels"
                    return
            except Exception as e:
                self.error_message = f"Invalid device index {self.device_index}: {str(e)}"
                return
            stream = audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )
            stream.start_stream()
            while self.is_recording:
                try:
                    audio_data = stream.read(CHUNK, exception_on_overflow=False)
                    if audio_data:
                        self.audio_frames.append(audio_data)
                    time.sleep(0.001)
                except Exception as e:
                    self.error_message = f"Error during recording: {str(e)}"
                    break
        except Exception as e:
            self.error_message = f"Recording setup error: {str(e)}"
        finally:
            try:
                if stream and stream.is_active():
                    stream.stop_stream()
                if stream:
                    stream.close()
            except:
                pass
            try:
                if audio_interface:
                    audio_interface.terminate()
            except:
                pass
    
    def _save_audio_file(self):
        if not self.audio_frames:
            return None
        try:
            # Use unique filename in temp directory
            temp_filename = f"recording_{uuid.uuid4().hex}.wav"
            temp_path = TEMP_DIR / temp_filename
            
            p = pyaudio.PyAudio()
            sample_width = p.get_sample_size(FORMAT)
            p.terminate()
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(RATE)
                wav_file.writeframes(b''.join(self.audio_frames))
            if temp_path.exists() and temp_path.stat().st_size > 44:
                self.temp_file = str(temp_path)
                return str(temp_path)
            return None
        except Exception as e:
            self.error_message = f"Error saving audio: {str(e)}"
            return None
    
    def cleanup(self):
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                self.temp_file = None
            except:
                pass

def post_process_transcription(text, custom_corrections=None):
    """Minimal post-processing to preserve actual speech patterns"""
    tweakspeare_variants = [
        "tweak spear", "tweak sphere", "tweak spare", "tweak spur",
        "tweet spear", "tweet sphere", "tweet spare", "tweak spire",
        "weak spear", "weak sphere", "weak spare", "twig spear",
        "tweak shakespeare", "tweet shakespeare", "tweek spear",
        "tweak pier", "tweak beer", "tweak peer", "tweak sheer",
        "tweaks peer", "tweaks spear", "tweaksphere", "tweak spiers",
        "tweak spears", "tweek sphere", "tweak sphear", "tweak spehere"
    ]
    corrected_text = text
    for variant in tweakspeare_variants:
        pattern = re.compile(re.escape(variant), re.IGNORECASE)
        corrected_text = pattern.sub("Tweakspeare", corrected_text)
    
    # Apply custom corrections
    if custom_corrections:
        for wrong, correct in custom_corrections.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            corrected_text = pattern.sub(correct, corrected_text)
    return corrected_text

def transcribe_audio_file(model, audio_file_path):
    """Enhanced transcription with settings to preserve actual speech"""
    try:
        if not os.path.exists(audio_file_path):
            return "Error: Audio file not found"
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            return "Error: Audio recording too short or empty"
        
        # Use specific Whisper parameters to get more accurate transcription
        result = model.transcribe(
            audio_file_path,
            language='en',
            verbose=False,
            fp16=False,
            temperature=0.0,  # More deterministic output
            beam_size=5,      # Better accuracy
            best_of=5,        # Multiple attempts for best result
            condition_on_previous_text=False  # Don't auto-correct based on context
        )
        
        transcribed_text = result["text"].strip()
        if not transcribed_text:
            return "No speech detected in the recording"
        
        # Minimal post-processing to preserve actual speech patterns
        corrected_text = post_process_transcription(transcribed_text)
        return corrected_text
    except Exception as e:
        return f"Transcription error: {str(e)}"

def check_errors(text, tool):
    """Enhanced error checking with categorization"""
    if not tool:
        return []
    try:
        matches = tool.check(text)
        errors = []
        for match in matches:
            error_type = categorize_error(match)
            errors.append({
                'error': match.context[match.offset:match.offset + match.errorLength],
                'message': match.message,
                'suggestions': match.replacements[:3],  # Limit to top 3 suggestions
                'start': match.offset,
                'end': match.offset + match.errorLength,
                'type': error_type,
                'severity': get_error_severity(match)
            })
        return errors
    except Exception as e:
        st.error(f"Error checking grammar: {str(e)}")
        return []

def categorize_error(match):
    """Categorize error types for better display"""
    rule_id = match.ruleId.lower()
    if 'spell' in rule_id or 'typo' in rule_id:
        return 'spelling'
    elif 'grammar' in rule_id or 'agreement' in rule_id:
        return 'grammar'
    elif 'punctuation' in rule_id:
        return 'punctuation'
    elif 'style' in rule_id or 'redundancy' in rule_id:
        return 'style'
    else:
        return 'other'

def get_error_severity(match):
    """Determine error severity"""
    if 'MORFOLOGIK_RULE' in match.ruleId:
        return 'high'  # Spelling errors
    elif any(word in match.ruleId.lower() for word in ['agreement', 'grammar']):
        return 'high'  # Grammar errors
    else:
        return 'medium'

def reframe_text_with_corrections(original_text, errors):
    """Generate a corrected version of the text"""
    if not errors:
        return original_text
    
    corrected_text = original_text
    # Sort errors by position (reverse order to maintain indices)
    sorted_errors = sorted(errors, key=lambda x: x['start'], reverse=True)
    
    for error in sorted_errors:
        if error['suggestions']:
            best_suggestion = error['suggestions'][0]
            start = error['start']
            end = error['end']
            corrected_text = corrected_text[:start] + best_suggestion + corrected_text[end:]
    
    return corrected_text

def generate_pronunciation(text):
    """Generate pronunciation with better error handling"""
    try:
        # Create unique filename to avoid permission issues
        pronunciation_file = TEMP_DIR / f"pronunciation_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(str(pronunciation_file))
        return str(pronunciation_file)
    except Exception as e:
        st.error(f"Error generating pronunciation: {str(e)}")
        return None

def play_pronunciation(audio_file):
    """Play pronunciation with better error handling"""
    try:
        # Initialize pygame mixer if not already done
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        # Clean up the file after playing
        try:
            os.remove(audio_file)
        except:
            pass
        
        return True
    except Exception as e:
        st.error(f"Error playing pronunciation: {str(e)}")
        return False

def display_highlighted_text(text, errors):
    """Display text with highlighted errors"""
    if not errors:
        return text
    
    # Create HTML version with highlighting
    html_text = text
    offset = 0
    
    # Sort errors by position
    sorted_errors = sorted(errors, key=lambda x: x['start'])
    
    for error in sorted_errors:
        start = error['start'] + offset
        end = error['end'] + offset
        error_text = error['error']
        
        # Color code by error type and severity
        color = get_error_color(error['type'], error['severity'])
        
        # Create highlighted span
        highlighted = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; color: white; font-weight: bold;" title="{error["message"]}">{error_text}</span>'
        
        html_text = html_text[:start] + highlighted + html_text[end:]
        offset += len(highlighted) - (end - start)
    
    return html_text

def get_error_color(error_type, severity):
    """Get color based on error type and severity"""
    colors = {
        'spelling': {'high': '#e74c3c', 'medium': '#e67e22'},
        'grammar': {'high': '#e74c3c', 'medium': '#f39c12'},
        'punctuation': {'high': '#9b59b6', 'medium': '#8e44ad'},
        'style': {'high': '#3498db', 'medium': '#5dade2'},
        'other': {'high': '#34495e', 'medium': '#7f8c8d'}
    }
    return colors.get(error_type, colors['other']).get(severity, '#7f8c8d')

def initialize_session_state():
    default_values = {
        "recorder": None,
        "transcription_result": "",
        "corrected_text": "",
        "is_currently_recording": False,
        "selected_mic_index": None,
        "recording_start_time": None,
        "transcription_history": [],
        "timer_placeholder": None,
        "custom_corrections": {},
        "show_corrections": True
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

def format_duration(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def update_timer():
    if st.session_state.is_currently_recording and st.session_state.recording_start_time:
        elapsed = time.time() - st.session_state.recording_start_time
        return format_duration(elapsed)
    return "00:00"

def display_transcription_stats(text):
    if not text or text.startswith("Error:") or text.startswith("No speech"):
        return
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", word_count)
    with col2:
        st.metric("Characters", char_count)
    with col3:
        st.metric("Sentences", max(sentence_count, 1))

def main():
    st.set_page_config(
        page_title="Tweakspeare - Speech Enhancement",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    if not check_dependencies():
        st.stop()
    model = load_model()
    tool = load_language_tool()
    if model is None or tool is None:
        st.stop()
    
    st.title("🎙️ Tweakspeare: AI-Powered Speech Enhancement")
    st.markdown("**Capture your actual speech patterns and get intelligent corrections**")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Select your microphone** from the dropdown below
        2. **Click 'Start Recording'** to begin capturing audio
        3. **Speak naturally** - don't worry about mistakes!
        4. **Click 'Stop Recording'** to see your actual speech
        5. **Review errors** and **hear correct pronunciations**
        6. **Practice** with the corrected version
        
        ---
        
        ### ⚙️ Advanced Settings:
        """)
        
        # Recording quality settings
        st.subheader("🎵 Recording Quality")
        quality_level = st.select_slider(
            "Quality Level:",
            options=["Basic", "Standard", "High"],
            value="Standard",
            help="Higher quality = better accuracy but slower processing"
        )
        
        st.subheader("🔧 Custom Corrections")
        if st.checkbox("Enable custom word corrections"):
            st.markdown("**Add words that are often misheard:**")
            col1, col2 = st.columns(2)
            with col1:
                wrong_word = st.text_input("Misheard as:", placeholder="e.g., 'john doe'")
            with col2:
                correct_word = st.text_input("Should be:", placeholder="e.g., 'Jon Doe'")
            if st.button("Add Correction") and wrong_word and correct_word:
                st.session_state.custom_corrections[wrong_word.lower()] = correct_word
                st.success(f"Added: '{wrong_word}' → '{correct_word}'")
            if st.session_state.custom_corrections:
                st.markdown("**Current corrections:**")
                for wrong, correct in st.session_state.custom_corrections.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"'{wrong}' → '{correct}'")
                    with col2:
                        if st.button("❌", key=f"del_{wrong}"):
                            del st.session_state.custom_corrections[wrong]
                            st.rerun()
        
        # Error display options
        st.subheader("📊 Error Display")
        show_severity = st.checkbox("Show error severity levels", value=True)
        show_categories = st.checkbox("Categorize error types", value=True)
        
        st.markdown("""
        ---
        
        ### 💡 Tips for Better Results:
        - **Speak naturally** - don't try to be perfect!
        - **Include common mistakes** - that's what we're here to fix
        - **Use normal conversational pace**
        - **Test different sentence structures**
        - **Practice difficult words multiple times**
        
        ### 🎨 Error Color Guide:
        - 🔴 **Red**: Spelling/Grammar (High Priority)
        - 🟠 **Orange**: Grammar (Medium Priority)  
        - 🟣 **Purple**: Punctuation Issues
        - 🔵 **Blue**: Style Suggestions
        - ⚫ **Gray**: Other Issues
        """)
    
    # Microphone selection
    microphones = list_microphones()
    if not microphones:
        st.error("❌ No microphones detected!")
        st.stop()
    
    st.subheader("🎤 Microphone Selection")
    mic_options = [f"{name} (Device {idx})" for idx, name in microphones]
    selected_mic_option = st.selectbox(
        "Choose your microphone:",
        mic_options
    )
    selected_device_idx = next(
        idx for idx, name in microphones 
        if f"{name} (Device {idx})" == selected_mic_option
    )
    
    if st.session_state.selected_mic_index != selected_device_idx:
        st.session_state.selected_mic_index = selected_device_idx
        if st.session_state.recorder:
            st.session_state.recorder.cleanup()
        st.session_state.recorder = AudioRecorder(selected_device_idx)
    
    # Recording controls
    st.subheader("🎙️ Recording Controls")
    control_col1, control_col2, status_col = st.columns([1, 1, 2])
    
    with control_col1:
        start_disabled = st.session_state.is_currently_recording
        if st.button("🔴 Start Recording", disabled=start_disabled, key="start_recording_btn"):
            if st.session_state.recorder:
                success = st.session_state.recorder.start_recording()
                if success:
                    st.session_state.is_currently_recording = True
                    st.session_state.recording_start_time = time.time()
                    st.rerun()
                else:
                    error = getattr(st.session_state.recorder, 'error_message', 'Unknown error')
                    st.error(f"Failed to start recording: {error}")
                    st.session_state.recorder = AudioRecorder(st.session_state.selected_mic_index)
    
    with control_col2:
        stop_disabled = not st.session_state.is_currently_recording
        if st.button("⏹️ Stop Recording", disabled=stop_disabled, key="stop_recording_btn"):
            if st.session_state.recorder and st.session_state.is_currently_recording:
                with st.spinner("🔄 Processing your speech..."):
                    result = st.session_state.recorder.stop_recording()
                    st.session_state.is_currently_recording = False
                    st.session_state.recording_start_time = None
                    
                    if result[0] is None:
                        error_msg = result[1] if result[1] else "Failed to process audio"
                        st.session_state.transcription_result = f"Error: {error_msg}"
                    else:
                        audio_file_path = result[0]
                        if os.path.exists(audio_file_path) and os.path.getsize(audio_file_path) > 1000:
                            transcription = transcribe_audio_file(model, audio_file_path)
                            st.session_state.transcription_result = transcription
                            
                            if not transcription.startswith("Error:") and not transcription.startswith("No speech"):
                                # Check for errors and generate corrected text
                                errors = check_errors(transcription, tool)
                                st.session_state.corrected_text = reframe_text_with_corrections(transcription, errors)
                                
                                st.session_state.transcription_history.append({
                                    "timestamp": time.strftime("%H:%M:%S"),
                                    "original": transcription,
                                    "corrected": st.session_state.corrected_text,
                                    "errors": len(errors)
                                })
                        else:
                            st.session_state.transcription_result = "Error: Audio too short or empty"
                        st.session_state.recorder.cleanup()
                    st.rerun()
    
    with status_col:
        if st.session_state.is_currently_recording:
            current_time = update_timer()
            st.markdown(f"🔴 **RECORDING... {current_time}**")
            time.sleep(1)
            st.rerun()
        else:
            st.empty()
    
    # Display results
    if st.session_state.transcription_result:
        if st.session_state.transcription_result.startswith("Error:"):
            st.error(st.session_state.transcription_result)
        elif st.session_state.transcription_result.startswith("No speech"):
            st.warning(st.session_state.transcription_result)
        else:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📝 Your Speech", "✅ Corrected Version", "📊 Analysis"])
            
            with tab1:
                st.subheader("📝 What You Actually Said")
                
                # Check for errors
                errors = check_errors(st.session_state.transcription_result, tool)
                
                if errors:
                    st.info(f"Found {len(errors)} areas for improvement")
                    
                    # Display highlighted text
                    highlighted_html = display_highlighted_text(st.session_state.transcription_result, errors)
                    st.markdown(highlighted_html, unsafe_allow_html=True)
                    
                    # Error details
                    st.subheader("🔍 Detailed Corrections")
                    for i, error in enumerate(errors):
                        with st.expander(f"{error['type'].title()} Error: '{error['error']}' ({error['severity']} priority)"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**Issue**: {error['message']}")
                                if error['suggestions']:
                                    st.write(f"**Suggestions**: {', '.join(error['suggestions'])}")
                                else:
                                    st.write("**No suggestions available**")
                            with col2:
                                if error['suggestions']:
                                    if st.button(f"🔊 Hear correct pronunciation", key=f"pron_{i}"):
                                        audio_file = generate_pronunciation(error['suggestions'][0])
                                        if audio_file:
                                            play_pronunciation(audio_file)
                else:
                    st.success("🎉 Perfect! No errors detected in your speech!")
                    st.text_area("Your speech:", value=st.session_state.transcription_result, height=120)
            
            with tab2:
                st.subheader("✅ Corrected Version")
                if st.session_state.corrected_text != st.session_state.transcription_result:
                    st.text_area("Corrected text:", value=st.session_state.corrected_text, height=120)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔊 Hear Original"):
                            audio_file = generate_pronunciation(st.session_state.transcription_result)
                            if audio_file:
                                play_pronunciation(audio_file)
                    with col2:
                        if st.button("🔊 Hear Corrected"):
                            audio_file = generate_pronunciation(st.session_state.corrected_text)
                            if audio_file:
                                play_pronunciation(audio_file)
                else:
                    st.success("Your speech was already perfect! No corrections needed.")
                    st.text_area("Your perfect speech:", value=st.session_state.transcription_result, height=120)
            
            with tab3:
                st.subheader("📊 Speech Analysis")
                display_transcription_stats(st.session_state.transcription_result)
                
                # Error breakdown
                errors = check_errors(st.session_state.transcription_result, tool)
                if errors:
                    error_types = {}
                    for error in errors:
                        error_types[error['type']] = error_types.get(error['type'], 0) + 1
                    
                    st.subheader("📈 Error Breakdown")
                    for error_type, count in error_types.items():
                        st.metric(f"{error_type.title()} Errors", count)
    
    # Session History
    if st.session_state.transcription_history:
        st.subheader("📚 Session History")
        with st.expander(f"View previous recordings ({len(st.session_state.transcription_history)} total)"):
            for i, entry in enumerate(reversed(st.session_state.transcription_history[-5:])):
                st.markdown(f"**{entry['timestamp']}** - {entry['errors']} errors found")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"*Original:* {entry['original'][:100]}...")
                with col2:
                    st.markdown(f"*Corrected:* {entry['corrected'][:100]}...")
                st.markdown("---")
        
        if st.button("🗑️ Clear History"):
            st.session_state.transcription_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*Enhanced with [OpenAI Whisper](https://openai.com/research/whisper) & [LanguageTool](https://languagetool.org/) • "
        "Built with [Streamlit](https://streamlit.io)*"
    )

def cleanup_on_exit():
    """Clean up temporary files on exit"""
    if hasattr(st.session_state, 'recorder') and st.session_state.recorder:
        st.session_state.recorder.cleanup()
    
    # Clean up temp directory
    try:
        for file in TEMP_DIR.glob("*.mp3"):
            file.unlink()
        for file in TEMP_DIR.glob("*.wav"):
            file.unlink()
    except:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_on_exit()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        cleanup_on_exit()