import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Real-Time AI Wallpaper AR", layout="centered")

st.title("📹 AR-Шпалери в режимі реального часу")
st.caption("Наведеніть камеру на стіну — шпалери накладаються на живому відеопотоці.")

# Конфігурація WebRTC (для стабільного підключення з мобільних пристроїв)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Налаштування у бічній панелі ---
st.sidebar.header("⚙️ Параметри маски")
sensitivity = st.sidebar.slider("Чутливість обтікання (Canny)", 10, 100, 35)
opacity = st.sidebar.slider("Прозорість шпалер", 0.1, 1.0, 0.75)
pattern_style = st.sidebar.selectbox("Стиль шпалер", ["Смарагд + Золото", "Цегла Лофт", "Текстурний Льон"])

# --- Функція генерації текстури ---
def get_pattern(style, h, w):
    size = 128
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    if style == "Смарагд + Золото":
        tile[:] = (43, 61, 16)
        cv2.circle(tile, (size//2, size//2), int(size*0.35), (89, 175, 212), 3)
    elif style == "Цегла Лофт":
        tile[:] = (46, 59, 122)
        cv2.rectangle(tile, (4, 4), (size-4, size//2-4), (27, 35, 74), -1)
    else:
        tile[:] = (210, 215, 218)
        for i in range(0, size, 8):
            cv2.line(tile, (i, 0), (i, size), (185, 190, 195), 1)
    
    rx = int(np.ceil(w / size)) + 1
    ry = int(np.ceil(h / size)) + 1
    tiled = np.tile(tile, (ry, rx, 1))
    return tiled[:h, :w]

# --- Обробка кожного кадру відеопотоку ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Отримання кадру з камери у форматі BGR (OpenCV)
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    # 1. Створення текстури шпалер
    texture = get_pattern(pattern_style, h, w)

    # 2. Обтікання об'єктів (Canny Edge detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, sensitivity, sensitivity * 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 3. Формування маски стіни
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[dilated_edges > 0] = 0

    # 4. Збереження освітлення
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, _, _ = cv2.split(lab)
    shadow_map = l_chan.astype(float) / 255.0
    shadow_map = np.clip(shadow_map * 1.15, 0, 1)

    # 5. Накладання шпалер
    textured_wall = (texture.astype(float) * shadow_map[:, :, np.newaxis]).astype(np.uint8)
    smooth_mask = cv2.GaussianBlur(mask, (7, 7), 0)[:, :, np.newaxis] / 255.0
    alpha = opacity * smooth_mask

    # Підсумковий кадр
    result = (img * (1 - alpha) + textured_wall * alpha).astype(np.uint8)

    return av.VideoFrame.from_ndarray(result, format="bgr24")

# --- Запуск WebRTC стриму (із захистом від відкриття програвача) ---
webrtc_streamer(
    key="wallpaper-ar-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"facingMode": "environment"},  # Намагається використати задню камеру
        "audio": False
    },
    video_html_attrs={
        "autoPlay": True,
        "controls": False,        # Приховує панель програвача (Play/Pause)
        "style": {"width": "100%"},
        "playsinline": True,      # Змушує відео грати всередині сторінки (не у плеєрі)
        "muted": True             # Обов'язково для автовідтворення на iOS/Android
    },
    async_processing=True,
)
