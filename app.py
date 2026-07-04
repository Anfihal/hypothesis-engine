import streamlit as st
import json
import os
import sys
import tempfile
import re
from datetime import datetime

# Импорт для OCR и PDF
try:
    from PIL import Image
    import pytesseract
except ImportError:
    pytesseract = None
    Image = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp.extractor import EntityExtractor
from src.graph.kgraph import KnowledgeGraph
from src.generation.deliberation import HypothesisDeliberation
from src.ranking.scorer import rank_hypotheses

# Настройка страницы
st.set_page_config(page_title="Генератор гипотез", page_icon="🧪", layout="wide")
st.title("🧪 Генератор научных гипотез")
st.markdown("---")

# --- Вспомогательные функции (OCR, PDF) ---
def extract_text_from_pdf(file):
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
        st.warning(f"Ошибка чтения PDF: {e}")
        return None

def extract_text_from_image(file):
    if pytesseract is None or Image is None:
        return None
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image, lang='eng+rus')
        return text.strip() if text.strip() else None
    except Exception as e:
        st.warning(f"Ошибка OCR: {e}")
        return None

def extract_text_from_pdf_with_ocr(file, poppler_path=None):
    if convert_from_path is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
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
        st.warning(f"Ошибка OCR PDF: {e}")
        return None

# --- Боковая панель (KPI, ограничения, модель) ---
with st.sidebar:
    st.header("Параметры")
    kpi = st.text_area(
        "Целевой KPI",
        value="Increase creep resistance of nickel-based superalloy by 20% at 700°C",
        help="Опишите желаемое свойство или проблему."
    )
    constraints = st.text_area(
        "Ограничения",
        value="No rhenium, budget < $500k, scalable to industrial production",
        help="Бюджет, сырьё, оборудование..."
    )
    st.subheader("Модель")
    use_yandex = st.checkbox("Использовать Yandex GPT (требуется API-ключ)", value=False)
    model_name = st.selectbox(
        "Локальная модель (Ollama)",
        options=["llama3.1:8b", "mistral:7b", "llama3.2:3b"],
        index=0
    ) if not use_yandex else None
    st.caption("Все вычисления локальные, данные не покидают ваш компьютер.")

# --- Основная область: текст + загрузка файлов ---
st.header("📄 Источники знаний (литература, отчёты, патенты)")

# Текстовое поле (всегда видимо)
literature_text = st.text_area(
    "Вставьте текст или отредактируйте пример:",
    value="""The addition of 0.5% niobium to Inconel 718 significantly increases yield strength at elevated temperatures.
Previous studies show that niobium forms stable carbides which hinder dislocation motion.
However, excess niobium may cause embrittlement due to phase precipitation.
Recent experiments indicate that a two-step aging treatment can further enhance creep resistance.
Also, the use of tantalum has been explored but is expensive and limited in supply.
Grain boundary engineering through thermomechanical processing improves durability.""",
    height=200
)

# Загрузка файлов (можно несколько)
uploaded_files = st.file_uploader(
    "Загрузите дополнительные файлы (txt, pdf, png, jpg)",
    type=["txt", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Обработка загруженных файлов
if uploaded_files:
    for file in uploaded_files:
        ext = file.name.split('.')[-1].lower()
        if ext == "txt":
            try:
                content = file.read().decode("utf-8")
                literature_text += "\n\n" + content
                st.success(f"Добавлен текст из {file.name}")
            except Exception as e:
                st.error(f"Ошибка чтения {file.name}: {e}")
        elif ext == "pdf":
            # Сначала пытаемся извлечь текст
            text_from_pdf = extract_text_from_pdf(file)
            if text_from_pdf and len(text_from_pdf) > 100:
                literature_text += "\n\n" + text_from_pdf
                st.success(f"Извлечён текст из PDF {file.name}")
            else:
                # Пробуем OCR
                file.seek(0)
                ocr_text = extract_text_from_pdf_with_ocr(file, poppler_path=None)
                if ocr_text:
                    literature_text += "\n\n" + ocr_text
                    st.success(f"Извлечён текст через OCR из {file.name}")
                else:
                    st.warning(f"Не удалось извлечь текст из {file.name}")
        elif ext in ["png", "jpg", "jpeg"]:
            file.seek(0)
            ocr_text = extract_text_from_image(file)
            if ocr_text:
                literature_text += "\n\n" + ocr_text
                st.success(f"Распознан текст из изображения {file.name}")
            else:
                st.warning(f"Не удалось распознать текст из {file.name}")

# --- Кнопка генерации ---
if st.button("🚀 Сгенерировать гипотезы", type="primary"):
    if not literature_text.strip():
        st.error("Пожалуйста, введите или загрузите текст литературы.")
    else:
        with st.spinner("Идёт генерация гипотез... (может занять 20–60 секунд)"):
            try:
                # 1. Извлечение сущностей и построение графа
                extractor = EntityExtractor()
                analysis = extractor.process_document(literature_text)
                
                kg = KnowledgeGraph()
                for ent in analysis['entities']:
                    kg.add_entity(ent['text'], ent['label'])
                for rel in analysis['relationships']:
                    kg.add_relation(rel['subject'], rel['relation'], rel['object'], rel['evidence'])
                
                graph_context = kg.to_text_context()
                
                # 2. Генерация гипотез
                delib = HypothesisDeliberation(
                    use_yandex=use_yandex,
                    model_name=model_name if not use_yandex else None
                )
                full_context = graph_context + "\nLiterature insights: " + "; ".join([
                    f"{rel['subject']} {rel['relation']} {rel['object']}"
                    for rel in analysis['relationships'][:5]
                ])
                
                hypotheses = delib.run(kpi, constraints, full_context, max_rounds=2)
                
                # 3. Ранжирование
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
                
                st.success(f"✅ Сгенерировано {len(ranked)} гипотез!")
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                st.exception(e)

# --- Отображение результатов (если есть) ---
if 'hypotheses' in st.session_state:
    st.markdown("---")
    st.header("📊 Результаты")
    
    # Граф знаний
    if 'kg' in st.session_state:
        with st.expander("🕸️ Граф знаний", expanded=False):
            kg = st.session_state['kg']
            if kg.graph.number_of_nodes() > 0:
                try:
                    from pyvis.network import Network
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
    
    # Список гипотез
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
        if st.button("📥 Скачать JSON"):
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
        if st.button("📄 Скачать TXT"):
            report = f"=== Отчёт ===\nKPI: {st.session_state.get('kpi', '')}\n"
            report += f"Ограничения: {st.session_state.get('constraints', '')}\n\n"
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