import streamlit as st
import json
import os
import sys
from datetime import datetime
import tempfile

# Импорт для работы с изображениями и OCR
try:
    from PIL import Image
    import pytesseract
except ImportError:
    st.error("Библиотеки для OCR не установлены. Выполните: pip install pytesseract pillow")
    pytesseract = None
    Image = None

try:
    import PyPDF2
except ImportError:
    st.error("Библиотека PyPDF2 не установлена. Выполните: pip install PyPDF2")
    PyPDF2 = None

# Для конвертации PDF в изображения (если нужно)
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp.extractor import EntityExtractor
from src.graph.kgraph import KnowledgeGraph
from src.generation.deliberation import HypothesisDeliberation
from src.ranking.scorer import rank_hypotheses

# Настройка страницы
st.set_page_config(
    page_title="Генератор гипотез для материаловедения",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Генератор научных гипотез")
st.markdown("---")

# --- Вспомогательные функции ---
def extract_text_from_pdf(file):
    """Извлекает текст из PDF-файла с использованием PyPDF2."""
    if PyPDF2 is None:
        return None
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text.strip() else None
    except Exception as e:
        st.warning(f"Не удалось извлечь текст из PDF: {e}")
        return None

def extract_text_from_image(image_file):
    """Извлекает текст из изображения с помощью Tesseract OCR."""
    if pytesseract is None or Image is None:
        return None
    try:
        image = Image.open(image_file)
        # Можно настроить язык: eng+rus для смешанных текстов
        text = pytesseract.image_to_string(image, lang='eng+rus')
        return text.strip() if text.strip() else None
    except Exception as e:
        st.warning(f"Ошибка OCR: {e}")
        return None

def extract_text_from_pdf_with_ocr(pdf_file, poppler_path=None):
    """
    Конвертирует PDF в изображения и применяет OCR к каждой странице.
    Требуется pdf2image и poppler.
    """
    if convert_from_path is None:
        return None
    try:
        # Сохраняем временно файл на диск (pdf2image работает с путём)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        images = convert_from_path(tmp_path, poppler_path=poppler_path, dpi=200)
        os.unlink(tmp_path)
        full_text = ""
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang='eng+rus')
            if text:
                full_text += f"--- Page {i+1} ---\n" + text + "\n"
        return full_text.strip() if full_text.strip() else None
    except Exception as e:
        st.warning(f"Ошибка при OCR PDF: {e}")
        return None

# Боковая панель для ввода параметров
with st.sidebar:
    st.header("Параметры")
    kpi = st.text_area(
        "Целевой KPI",
        value="Increase creep resistance of nickel-based superalloy by 20% at 700°C",
        help="Опишите желаемое свойство или технологическую проблему."
    )
    constraints = st.text_area(
        "Ограничения",
        value="No rhenium, budget < $500k, scalable to industrial production",
        help="Укажите доступные ресурсы, бюджет, оборудование и т.д."
    )
    
    st.subheader("Модель")
    use_yandex = st.checkbox("Использовать Yandex GPT (требуется API-ключ)", value=False)
    model_name = st.selectbox(
        "Локальная модель (Ollama)",
        options=["llama3.1:8b", "mistral:7b", "llama3.2:3b"],
        index=0
    ) if not use_yandex else None

    st.markdown("---")
    st.caption("Все вычисления производятся локально. Данные не передаются вовне.")

# Основная область
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📄 Источники знаний")
    input_method = st.radio(
        "Выберите способ ввода литературы:",
        ["Вставить текст", "Загрузить файл"]
    )
    
    literature_text = ""
    if input_method == "Вставить текст":
        literature_text = st.text_area(
            "Вставьте текст научных статей, патентов или отчётов:",
            value="""
The addition of 0.5% niobium to Inconel 718 significantly increases yield strength at elevated temperatures.
Previous studies show that niobium forms stable carbides which hinder dislocation motion.
However, excess niobium may cause embrittlement due to phase precipitation.
Recent experiments indicate that a two-step aging treatment can further enhance creep resistance.
Also, the use of tantalum has been explored but is expensive and limited in supply.
Grain boundary engineering through thermomechanical processing improves durability.
""",
            height=200
        )
    else:
        uploaded_file = st.file_uploader(
            "Загрузите текстовый файл (.txt), PDF (.pdf) или изображение (.png, .jpg, .jpeg)",
            type=["txt", "pdf", "png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == "txt":
                literature_text = uploaded_file.read().decode("utf-8")
                st.success("Текстовый файл загружен.")
            
            elif file_extension == "pdf":
                # Сначала пытаемся извлечь текст через PyPDF2
                text_from_pdf = extract_text_from_pdf(uploaded_file)
                if text_from_pdf and len(text_from_pdf) > 100:
                    literature_text = text_from_pdf
                    st.success(f"Извлечено {len(literature_text)} символов текста из PDF.")
                else:
                    # Если текста мало (или нет), пробуем OCR через конвертацию в изображения
                    st.info("В PDF мало текста. Пытаемся распознать через OCR...")
                    # Сбрасываем указатель файла (после чтения PyPDF2)
                    uploaded_file.seek(0)
                    # Параметр poppler_path укажите, если не добавлен в PATH
                    # Пример: poppler_path = r"C:\poppler\bin"
                    ocr_text = extract_text_from_pdf_with_ocr(uploaded_file, poppler_path=None)
                    if ocr_text:
                        literature_text = ocr_text
                        st.success(f"Извлечено {len(literature_text)} символов через OCR.")
                    else:
                        st.error("Не удалось извлечь текст из PDF. Возможно, файл защищён или не содержит текста.")
            
            elif file_extension in ["png", "jpg", "jpeg"]:
                # Распознаём текст на изображении
                uploaded_file.seek(0)
                ocr_text = extract_text_from_image(uploaded_file)
                if ocr_text:
                    literature_text = ocr_text
                    st.success(f"Извлечено {len(literature_text)} символов из изображения.")
                else:
                    st.error("Не удалось распознать текст на изображении.")
            
            if literature_text:
                st.text_area("Содержимое файла:", literature_text, height=200)

    if st.button("🚀 Сгенерировать гипотезы", type="primary"):
        if not literature_text.strip():
            st.error("Пожалуйста, введите или загрузите текст литературы.")
        else:
            with st.spinner("Идёт генерация гипотез... Это может занять 15–40 секунд."):
                try:
                    # 1. Извлечение сущностей
                    extractor = EntityExtractor()
                    analysis = extractor.process_document(literature_text)
                    
                    # 2. Построение графа знаний
                    kg = KnowledgeGraph()
                    for ent in analysis['entities']:
                        kg.add_entity(ent['text'], ent['label'])
                    for rel in analysis['relationships']:
                        kg.add_relation(rel['subject'], rel['relation'], rel['object'], rel['evidence'])
                    
                    graph_context = kg.to_text_context()
                    
                    # 3. Генерация гипотез
                    delib = HypothesisDeliberation(
                        use_yandex=use_yandex,
                        model_name=model_name if not use_yandex else None
                    )
                    full_context = graph_context + "\nLiterature insights: " + "; ".join([
                        f"{rel['subject']} {rel['relation']} {rel['object']}" 
                        for rel in analysis['relationships'][:5]
                    ])
                    
                    hypotheses = delib.run(kpi, constraints, full_context, max_rounds=2)
                    
                    # 4. Ранжирование
                    for hyp in hypotheses:
                        if 'novelty' not in hyp:
                            hyp['novelty'] = 0.7
                        if 'impact' not in hyp:
                            hyp['impact'] = 0.6
                        if 'risk' not in hyp:
                            hyp['risk'] = 0.3
                        if isinstance(hyp.get('risk'), str):
                            risk_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
                            hyp['risk'] = risk_map.get(hyp['risk'].lower(), 0.5)
                        if isinstance(hyp.get('impact'), str):
                            import re
                            nums = re.findall(r'\d+', hyp['impact'])
                            if nums:
                                hyp['impact'] = float(nums[0]) / 100.0
                            else:
                                hyp['impact'] = 0.6
                        if isinstance(hyp.get('novelty'), str):
                            hyp['novelty'] = 0.7 if 'novel' in hyp['novelty'].lower() else 0.5
                    
                    ranked = rank_hypotheses(hypotheses)
                    
                    # Сохраняем в сессию
                    st.session_state['hypotheses'] = ranked
                    st.session_state['kg'] = kg
                    st.session_state['kpi'] = kpi
                    st.session_state['constraints'] = constraints
                    st.session_state['analysis'] = analysis
                    
                    st.success(f"✅ Сгенерировано {len(ranked)} гипотез!")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    st.exception(e)

# Вывод результатов (остаётся без изменений)
if 'hypotheses' in st.session_state:
    st.markdown("---")
    st.header("📊 Результаты")
    
    # Визуализация графа (если есть)
    if 'kg' in st.session_state:
        with st.expander("🕸️ Граф знаний", expanded=False):
            kg = st.session_state['kg']
            if kg.graph.number_of_nodes() > 0:
                try:
                    from pyvis.network import Network
                    import tempfile
                    net = Network(height="400px", width="100%")
                    for node in kg.graph.nodes:
                        net.add_node(node, label=node, title=kg.graph.nodes[node].get('type', ''))
                    for u, v, data in kg.graph.edges(data=True):
                        net.add_edge(u, v, label=data.get('relation', ''))
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                        net.save_graph(tmp.name)
                        with open(tmp.name, 'r', encoding='utf-8') as f:
                            html = f.read()
                        st.components.v1.html(html, height=450)
                    os.unlink(tmp.name)
                except Exception as e:
                    st.warning(f"Не удалось отобразить граф: {e}")
            else:
                st.info("Граф пуст.")
    
    # Таблица гипотез
    st.subheader("📋 Ранжированные гипотезы")
    hypotheses = st.session_state['hypotheses']
    
    for i, hyp in enumerate(hypotheses, 1):
        score = hyp.get('score', 0)
        color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
        with st.container():
            st.markdown(f"### {color} Гипотеза {i} (Score: {score:.2f})")
            st.markdown(f"**Заявление:** {hyp.get('statement', 'N/A')}")
            st.markdown(f"**Механизм:** {hyp.get('mechanism', 'N/A')}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Новизна", f"{hyp.get('novelty', 0):.2f}")
            col2.metric("Влияние", f"{hyp.get('impact', 0):.2f}")
            col3.metric("Риск", f"{hyp.get('risk', 0):.2f}")
            st.markdown("---")
    
    # Экспорт
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Скачать результаты (JSON)"):
            output = {
                "kpi": st.session_state.get('kpi', ''),
                "constraints": st.session_state.get('constraints', ''),
                "timestamp": datetime.now().isoformat(),
                "hypotheses": hypotheses
            }
            json_str = json.dumps(output, indent=2, ensure_ascii=False)
            st.download_button(
                label="Скачать JSON",
                data=json_str,
                file_name=f"hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("📄 Скачать отчёт (TXT)"):
            report = f"=== Отчёт по генерации гипотез ===\n"
            report += f"KPI: {st.session_state.get('kpi', '')}\n"
            report += f"Ограничения: {st.session_state.get('constraints', '')}\n"
            report += f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            for i, hyp in enumerate(hypotheses, 1):
                report += f"Гипотеза {i} (Score: {hyp.get('score', 0):.2f})\n"
                report += f"  Заявление: {hyp.get('statement', 'N/A')}\n"
                report += f"  Механизм: {hyp.get('mechanism', 'N/A')}\n"
                report += f"  Новизна: {hyp.get('novelty', 0):.2f}\n"
                report += f"  Влияние: {hyp.get('impact', 0):.2f}\n"
                report += f"  Риск: {hyp.get('risk', 0):.2f}\n\n"
            st.download_button(
                label="Скачать TXT",
                data=report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )