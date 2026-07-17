import os
import re
import cv2
import tempfile
import yt_dlp
import streamlit as st
from PIL import Image
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

# --- UI Setup ---
st.set_page_config(page_title="YouTube Note Takeaway", page_icon="📹", layout="centered")
st.title("📹 YouTube Video to PDF & Transcript")
st.write("Extract visual slides and download transcripts from any YouTube video directly in your browser.")

# --- Helper Functions ---
def extract_video_id(url):
    pattern = r'(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/|e\/|watch\?v=)([^#\&\?]*);?'
    match = re.search(pattern, url)
    if match and len(match.group(1)) == 11:
        return match.group(1)
    return None

def format_timestamp(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# --- Core Logic Functions ---
def download_youtube_video(video_url, destination_dir, cookie_path=None):
    # This selector fetches mp4 specifically up to 720p, fallback to best overall if not found.
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(destination_dir, 'input_video.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    
    if cookie_path:
        ydl_opts['cookiefile'] = cookie_path
        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        return ydl.prepare_filename(info)

def fetch_transcript_with_proxy(video_url, proxy_type=None, proxy_user=None, proxy_pass=None, proxy_http=None, proxy_https=None):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Error: Invalid YouTube URL."
    
    try:
        proxy_config = None
        
        # Configure the proxy if inputs are provided in the Advanced tab
        if proxy_type == "Webshare" and proxy_user and proxy_pass:
            proxy_config = WebshareProxyConfig(
                proxy_username=proxy_user,
                proxy_password=proxy_pass
            )
        elif proxy_type == "Generic HTTP/S" and (proxy_http or proxy_https):
            proxy_config = GenericProxyConfig(
                http_url=proxy_http if proxy_http else None,
                https_url=proxy_https if proxy_https else None
            )

        # Correct instantiation to support the latest library APIs
        if proxy_config:
            api_instance = YouTubeTranscriptApi(proxy_config=proxy_config)
        else:
            api_instance = YouTubeTranscriptApi()
            
        priority_languages = ['hi-Latn', 'hi', 'en', 'en-IN']
        try:
            transcript_list = api_instance.list(video_id)
            transcript = transcript_list.find_transcript(priority_languages)
        except Exception:
            transcript_list = api_instance.list(video_id)
            transcript = transcript_list.find_default_transcript()
        
        transcript_data = transcript.fetch()
        return "\n".join([entry['text'].replace('\n', ' ') for entry in transcript_data])
    except Exception as e:
        return (f"Could not fetch transcript. (Captions might be disabled or Cloud IP is blocked).\n"
                f"Details: {e}")

def extract_slides_to_pdf(video_path, output_pdf_path, threshold, jpeg_quality):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video file."

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
        
    frame_interval = max(1, round(video_fps))
    
    temp_img_dir = tempfile.TemporaryDirectory()
    saved_frame_paths = []
    last_saved_fingerprint = None
    frame_count = 0
    extracted_count = 0

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        if frame_count % frame_interval != 0:
            if not cap.grab(): break
            frame_count += 1
            continue

        ret, frame = cap.retrieve()
        if not ret:
            ret, frame = cap.read()
            if not ret: break

        current_second = frame_count / video_fps
        timestamp_str = format_timestamp(current_second)

        # Slide change fingerprint check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)

        should_save = False
        if last_saved_fingerprint is not None:
            diff = cv2.absdiff(blurred, last_saved_fingerprint)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            change_percentage = (cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])) * 100
            if change_percentage > threshold:
                should_save = True
        else:
            should_save = True

        if should_save:
            height, width = frame.shape[:2]
            box_width = int(width * 0.12) if width > 1000 else 115
            box_height = int(height * 0.04) if height > 600 else 30
            font_scale = 0.5 * (height / 540)
            thickness = max(1, int(font_scale * 1.5))

            cv2.rectangle(frame, (5, 5), (box_width, box_height), (0, 0, 0), -1)
            cv2.putText(frame, timestamp_str, (int(box_width * 0.1), int(box_height * 0.7)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            temp_filename = f"slide_{extracted_count:05d}.jpg"
            temp_frame_path = os.path.join(temp_img_dir.name, temp_filename)
            cv2.imwrite(temp_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            
            saved_frame_paths.append(temp_frame_path)
            extracted_count += 1
            last_saved_fingerprint = blurred

        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(1.0, frame_count / total_frames))
            status_text.text(f"Processing frame timestamp: {timestamp_str} | Captured Slides: {extracted_count}")

    cap.release()
    progress_bar.empty()
    status_text.empty()

    if not saved_frame_paths:
        temp_img_dir.cleanup()
        return False, "No unique slides found based on your threshold settings."

    # Dynamic Page Layout with FPDF (highly memory-efficient)
    pdf = FPDF()
    for img_path in saved_frame_paths:
        with Image.open(img_path) as img:
            w, h = img.size
        orientation = 'L' if w > h else 'P'
        pdf.add_page(orientation=orientation, format=(h, w) if orientation == 'L' else (w, h))
        pdf.image(img_path, x=0, y=0, w=pdf.w, h=pdf.h)
    
    pdf.output(output_pdf_path)
    temp_img_dir.cleanup()
    return True, f"Success! Rendered PDF containing {extracted_count} slides."

# --- UI Layout ---
url_input = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

with st.sidebar:
    st.header("⚙️ Slide Settings")
    threshold = st.slider("Slide Sensitivity (Higher = fewer duplicates captured)", 1.0, 10.0, 3.0, 0.5)
    jpeg_quality = st.slider("JPEG Quality (Higher = crispier slides, larger PDF size)", 50, 100, 80, 5)
    
    st.markdown("---")
    st.header("🛡️ Cloud IP-Bypass Options")
    st.markdown("If the cloud provider's IP gets blocked by YouTube, use the parameters below:")
    
    cookie_file = st.file_uploader("Upload cookies.txt file", type=["txt"])
    
    st.markdown("**Proxy Settings**")
    p_type = st.selectbox("Proxy Config Type", ["None", "Webshare", "Generic HTTP/S"])
    
    p_user, p_pass = None, None
    p_http, p_https = None, None
    
    if p_type == "Webshare":
        p_user = st.text_input("Webshare Username")
        p_pass = st.text_input("Webshare Password", type="password")
    elif p_type == "Generic HTTP/S":
        p_http = st.text_input("HTTP Proxy URL (e.g., http://user:pass@host:port)")
        p_https = st.text_input("HTTPS Proxy URL (e.g., https://user:pass@host:port)")

if url_input:
    if st.button("Process Video", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = None
            if cookie_file:
                cookie_path = os.path.join(tmpdir, "cookies.txt")
                with open(cookie_path, "wb") as f:
                    f.write(cookie_file.getbuffer())
            
            # 1. Process and Fetch Transcript
            st.subheader("📝 Transcript Result")
            with st.spinner("Fetching transcript..."):
                transcript_text = fetch_transcript_with_proxy(
                    url_input, 
                    proxy_type=p_type, 
                    proxy_user=p_user, 
                    proxy_pass=p_pass, 
                    proxy_http=p_http, 
                    proxy_https=p_https
                )
                
            if "Error" not in transcript_text and "Could not fetch" not in transcript_text:
                st.success("Transcript compiled successfully!")
                st.download_button("📥 Download Transcript (.txt)", data=transcript_text, file_name="transcript.txt", mime="text/plain")
            else:
                st.warning(transcript_text)
                st.info("💡 Tip: If you are seeing an IP Block error, toggle the 'Cloud IP-Bypass Options' in the sidebar.")
            
            st.markdown("---")
            
            # 2. Process and Fetch Slides
            st.subheader("🖼️ Visual Slide Extraction")
            try:
                with st.spinner("Downloading video content (Hold tight)..."):
                    video_file_path = download_youtube_video(url_input, tmpdir, cookie_path)
                
                pdf_output_path = os.path.join(tmpdir, "extracted_slides.pdf")
                
                with st.spinner("Analyzing frames and writing PDF pages..."):
                    success, message = extract_slides_to_pdf(video_file_path, pdf_output_path, threshold, jpeg_quality)
                
                if success:
                    st.success(message)
                    with open(pdf_output_path, "rb") as f:
                        st.download_button("📥 Download Slides PDF", data=f.read(), file_name="presentation_slides.pdf", mime="application/pdf")
                else:
                    st.error(message)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
