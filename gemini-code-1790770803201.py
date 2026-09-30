import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Налаштування мобільного інтерфейсу
st.set_page_config(
    page_title="AI Wallpaper Fitting",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Стилізація для мобільних пристроїв
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007aff; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ AI Примірка Шпалер")
st.caption("Автоматичне маскування стін з обтіканням об'єктів та збереженням тіней")

# --- Алгоритми обробки (OpenCV + AI Masking) ---

def generate_procedural_texture(style_name, width, height, tile_scale=1.0):
    """Генерація реалістичних текстур шпалер"""
    size = int(128 * tile_scale)
    tile = np.zeros((size, size, 3), dtype=np.uint8)

    if style_name == "Смарагд + Золото":
        tile[:] = (43, 61, 16) # BGR
        cv2.circle(tile, (size//2, size//2), int(size*0.35), (89, 175, 212), 3)
    elif style_name == "Цегла Лофт":
        tile[:] = (46, 59, 122)
        cv2.rectangle(tile, (4, 4), (size-4, size//2-4), (27, 35, 74), -1)
        cv2.rectangle(tile, (4, size//2+4), (size-4, size-4), (27, 35, 74), -1)
    elif style_name == "Текстурний Льон":
        tile[:] = (210, 215, 218)
        for i in range(0, size, 6):
            cv2.line(tile, (i, 0), (i, size), (185, 190, 195), 1)
            cv2.line(tile, (0, i), (size, i), (185, 190, 195), 1)
    
    repeats_x = int(np.ceil(width / size)) + 1
    repeats_y = int(np.ceil(height / size)) + 1
    full_pattern = np.tile(tile, (repeats_y, repeats_x, 1))
    return full_pattern[:height, :width]


def process_wall_mask(room_bgr, texture_bgr, sensitivity, opacity):
    """Обтікання меблів/рослин та збереження тіней"""
    h, w, _ = room_bgr.shape
    texture_resized = cv2.resize(texture_bgr, (w, h))

    # 1. Виявлення контурів переднього плану (меблі, рослини)
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Детектор меж Canny + розширення для створення безпечної зони
    edges = cv2.Canny(blur, sensitivity, sensitivity * 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # 2. Маска стіни (виключає меблі та рослини)
    wall_mask = np.ones((h, w), dtype=np.uint8) * 255
    wall_mask[dilated_edges > 0] = 0

    # 3. Карта природного освітлення (LAB L-channel)
    lab = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    shadow_map = l_channel.astype(float) / 255.0
    shadow_map = np.clip(shadow_map * 1.15, 0, 1)

    # 4. Змішування текстури з тінями стіни
    textured_wall = (texture_resized.astype(float) * shadow_map[:, :, np.newaxis]).astype(np.uint8)

    # 5. Фінальне накладання з розмиттям меж маски
    smooth_mask = cv2.GaussianBlur(wall_mask, (7, 7), 0)[:, :, np.newaxis] / 255.0
    alpha = opacity * smooth_mask

    output_bgr = (room_bgr * (1 - alpha) + textured_wall * alpha).astype(np.uint8)
    return output_bgr


# --- Меню налаштувань ---
with st.expander("⚙️ Налаштування ШІ та Шпалер", expanded=False):
    wp_choice = st.selectbox("Обрати текстуру:", ["Смарагд + Золото", "Цегла Лофт", "Текстурний Льон", "Завантажити свою"])
    
    custom_texture = None
    if wp_choice == "Завантажити свою":
        custom_file = st.file_uploader("Файл шпалер", type=["jpg", "png"])
        if custom_file:
            custom_texture = Image.open(custom_file)

    sens = st.slider("Чутливість обтікання об'єктів", 10, 80, 30)
    opac = st.slider("Прозорість / Насиченість", 0.2, 1.0, 0.8)
    scale = st.slider("Масштаб малюнка", 0.5, 2.5, 1.0)


# --- Отримання зображення з Android ---
source_mode = st.radio("Джерело фото:", ["📷 Камера смартфона", "📁 Галерея"], horizontal=True)

img_data = None
if source_mode == "📷 Камера смартфона":
    img_data = st.camera_input("Зробіть фото стіни з кімнати")
else:
    uploaded = st.file_uploader("Завантажте фото кімнати з галереї", type=["jpg", "png", "jpeg"])
    if uploaded:
        img_data = uploaded


# --- Обробка та результат ---
if img_data is not None:
    # Завантаження фото
    room_pil = Image.open(img_data)
    room_bgr = cv2.cvtColor(np.array(room_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w, _ = room_bgr.shape

    # Підготовка текстури
    if custom_texture is not None:
        tex_rgb = np.array(custom_texture.convert('RGB'))
        texture_bgr = cv2.cvtColor(tex_rgb, cv2.COLOR_RGB2BGR)
    else:
        texture_bgr = generate_procedural_texture(wp_choice, w, h, scale)

    # Запуск AI-обробки
    with st.spinner("AI обробляє фото та накладає шпалери..."):
        res_bgr = process_wall_mask(room_bgr, texture_bgr, sens, opac)
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("✨ Результат примірки:")
    st.image(res_rgb, use_container_width=True)

    # Завантаження готового фото
    result_pil = Image.fromarray(res_rgb)
    import io
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    st.download_button("💾 Зберегти фото на телефон", data=buf.getvalue(), file_name="wallpaper_ar.jpg", mime="image/jpeg")