import streamlit as st
import json
import os
import sys
import tempfile
import re
from datetime import datetime

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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nlp.extractor import EntityExtractor
from src.graph.kgraph import KnowledgeGraph
from src.generation.deliberation import HypothesisDeliberation
from src.ranking.scorer import rank_hypotheses

# --- Словари для мультиязычного интерфейса ---
TEXTS = {
    'en': {
        'title': "🧪 Scientific Hypothesis Generator",
        'params': "Parameters",
        'kpi': "Target KPI",
        'constraints': "Constraints",
        'model': "Model",
        'use_yandex': "Use Yandex GPT (fallback)",
        'local_model': "Local model (Ollama)",
        'lang': "Language",
        'sources': "Knowledge Sources",
        'text_input': "Paste text or edit the example:",
        'upload': "Upload additional files (txt, pdf, png, jpg)",
        'generate': "🚀 Generate Hypotheses",
        'results': "📊 Results",
        'graph': "🕸️ Knowledge Graph",
        'score': "Score",
        'statement': "Statement",
        'mechanism': "Mechanism",
        'novelty': "Novelty",
        'risk': "Risk",
        'impact': "Impact",
        'sources_label': "Sources",
        'explanation': "Explanation",
        'recommendation': "Recommendation",
        'time': "Generation time",
        'export_json': "📥 Download JSON",
        'export_txt': "📄 Download TXT",
        'success': "✅ Generated {count} hypotheses!",
        'error_no_text': "Please enter or upload literature text.",
        'error_general': "❌ Error: {error}",
        'empty_graph': "Graph is empty."
    },
    'ru': {
        'title': "🧪 Генератор научных гипотез",
        'params': "Параметры",
        'kpi': "Целевой KPI",
        'constraints': "Ограничения",
        'model': "Модель",
        'use_yandex': "Использовать Yandex GPT (резерв)",
        'local_model': "Локальная модель (Ollama)",
        'lang': "Язык",
        'sources': "Источники знаний",
        'text_input': "Вставьте текст или отредактируйте пример:",
        'upload': "Загрузите дополнительные файлы (txt, pdf, png, jpg)",
        'generate': "🚀 Сгенерировать гипотезы",
        'results': "📊 Результаты",
        'graph': "🕸️ Граф знаний",
        'score': "Оценка",
        'statement': "Заявление",
        'mechanism': "Механизм",
        'novelty': "Новизна",
        'risk': "Риск",
        'impact': "Влияние",
        'sources_label': "Источники",
        'explanation': "Объяснение",
        'recommendation': "Рекомендация",
        'time': "Время генерации",
        'export_json': "📥 Скачать JSON",
        'export_txt': "📄 Скачать TXT",
        'success': "✅ Сгенерировано {count} гипотез!",
        'error_no_text': "Пожалуйста, введите или загрузите текст литературы.",
        'error_general': "❌ Ошибка: {error}",
        'empty_graph': "Граф пуст."
    },
    'zh': {
        'title': "🧪 科学假设生成器",
        'params': "参数",
        'kpi': "目标 KPI",
        'constraints': "约束条件",
        'model': "模型",
        'use_yandex': "使用 Yandex GPT (备用)",
        'local_model': "本地模型 (Ollama)",
        'lang': "语言",
        'sources': "知识来源",
        'text_input': "粘贴文本或编辑示例：",
        'upload': "上传附加文件 (txt, pdf, png, jpg)",
        'generate': "🚀 生成假设",
        'results': "📊 结果",
        'graph': "🕸️ 知识图谱",
        'score': "评分",
        'statement': "陈述",
        'mechanism': "机制",
        'novelty': "新颖性",
        'risk': "风险",
        'impact': "影响",
        'sources_label': "来源",
        'explanation': "解释",
        'recommendation': "建议",
        'time': "生成时间",
        'export_json': "📥 下载 JSON",
        'export_txt': "📄 下载 TXT",
        'success': "✅ 已生成 {count} 个假设！",
        'error_no_text': "请输入或上传文献文本。",
        'error_general': "❌ 错误：{error}",
        'empty_graph': "图谱为空。"
    }
}

st.set_page_config(page_title="Генератор гипотез", page_icon="🧪", layout="wide")

# --- Выбор языка ---
lang = st.sidebar.selectbox("Language / Язык / 语言", ["en", "ru", "zh"], index=1)
t = TEXTS[lang]

st.title(t['title'])
st.markdown("---")

# --- Вспомогательные функции ---
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

# --- Боковая панель ---
with st.sidebar:
    st.header(t['params'])
    kpi = st.text_area(t['kpi'], value="Increase creep resistance of nickel-based superalloy by 20% at 700°C")
    constraints = st.text_area(t['constraints'], value="No rhenium, budget < $500k, scalable to industrial production")
    st.subheader(t['model'])
    use_yandex = st.checkbox(t['use_yandex'], value=False)
    model_name = st.selectbox(t['local_model'], ["llama3.1:8b", "mistral:7b", "llama3.2:3b"], index=0) if not use_yandex else None
    st.caption("По умолчанию используется локальная модель. При недоступности – автоматический fallback на Yandex (если задан ключ).")

# --- Основная область ---
st.header(t['sources'])
literature_text = st.text_area(
    t['text_input'],
    value="""The addition of 0.5% niobium to Inconel 718 significantly increases yield strength at elevated temperatures.
Previous studies show that niobium forms stable carbides which hinder dislocation motion.
However, excess niobium may cause embrittlement due to phase precipitation.
Recent experiments indicate that a two-step aging treatment can further enhance creep resistance.
Also, the use of tantalum has been explored but is expensive and limited in supply.
Grain boundary engineering through thermomechanical processing improves durability.""",
    height=200
)

uploaded_files = st.file_uploader(t['upload'], type=["txt", "pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

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
            text_from_pdf = extract_text_from_pdf(file)
            if text_from_pdf and len(text_from_pdf) > 100:
                literature_text += "\n\n" + text_from_pdf
                st.success(f"Извлечён текст из PDF {file.name}")
            else:
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
if st.button(t['generate'], type="primary"):
    if not literature_text.strip():
        st.error(t['error_no_text'])
    else:
        with st.spinner("Идёт генерация гипотез... (может занять 20–60 секунд)"):
            try:
                extractor = EntityExtractor()
                analysis = extractor.process_document(literature_text)
                kg = KnowledgeGraph()
                for ent in analysis['entities']:
                    kg.add_entity(ent['text'], ent['label'])
                for rel in analysis['relationships']:
                    kg.add_relation(rel['subject'], rel['relation'], rel['object'], rel['evidence'])
                graph_context = kg.to_text_context()
                
                delib = HypothesisDeliberation(use_yandex=use_yandex, model_name=model_name if not use_yandex else "llama3.1:8b", timeout=45)
                full_context = graph_context + "\nLiterature insights: " + "; ".join([f"{rel['subject']} {rel['relation']} {rel['object']}" for rel in analysis['relationships'][:5]])
                
                hypotheses = delib.run(kpi, constraints, full_context, language=lang, max_rounds=2)
                
                # Ранжирование
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
                    hyp.setdefault('sources', [])
                    hyp.setdefault('explanation', '')
                    hyp.setdefault('recommendation', '')
                
                ranked = rank_hypotheses(hypotheses)
                st.session_state['hypotheses'] = ranked
                st.session_state['kg'] = kg
                st.session_state['kpi'] = kpi
                st.session_state['constraints'] = constraints
                st.success(t['success'].format(count=len(ranked)))
                
            except Exception as e:
                st.error(t['error_general'].format(error=str(e)))
                st.exception(e)

# --- Отображение результатов ---
if 'hypotheses' in st.session_state:
    st.markdown("---")
    st.header(t['results'])
    
    if 'kg' in st.session_state:
        with st.expander(t['graph'], expanded=False):
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
                st.info(t['empty_graph'])
    
    hypotheses = st.session_state['hypotheses']
    for i, hyp in enumerate(hypotheses, 1):
        score = hyp.get('score', 0)
        color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
        with st.container():
            st.markdown(f"### {color} {t['statement']} {i} ({t['score']}: {score:.2f})")
            st.markdown(f"**{t['statement']}:** {hyp.get('statement', 'N/A')}")
            st.markdown(f"**{t['mechanism']}:** {hyp.get('mechanism', 'N/A')}")
            st.markdown(f"**{t['novelty']}:** {hyp.get('novelty', 0):.2f}")
            st.markdown(f"**{t['risk']}:** {hyp.get('risk', 0):.2f}")
            st.markdown(f"**{t['impact']}:** {hyp.get('impact', 0):.2f}")
            sources = hyp.get('sources', [])
            if sources:
                st.markdown(f"**{t['sources_label']}:** {', '.join(sources)}")
            else:
                st.markdown(f"**{t['sources_label']}:** {t.get('not specified', 'не указаны')}")
            st.markdown(f"**{t['explanation']}:** {hyp.get('explanation', 'N/A')}")
            st.markdown(f"**{t['recommendation']}:** {hyp.get('recommendation', 'N/A')}")
            if hyp.get('generation_time'):
                st.caption(f"⏱️ {t['time']}: {hyp['generation_time']} сек")
            st.markdown("---")
    
    # Экспорт
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t['export_json']):
            output = {
                "kpi": st.session_state.get('kpi', ''),
                "constraints": st.session_state.get('constraints', ''),
                "language": lang,
                "timestamp": datetime.now().isoformat(),
                "hypotheses": hypotheses
            }
            json_str = json.dumps(output, indent=2, ensure_ascii=False)
            st.download_button(label="Скачать JSON", data=json_str, file_name=f"hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
    with col2:
        if st.button(t['export_txt']):
            report = f"=== {t['title']} ===\n"
            report += f"{t['kpi']}: {st.session_state.get('kpi', '')}\n"
            report += f"{t['constraints']}: {st.session_state.get('constraints', '')}\n"
            report += f"Language: {lang}\n\n"
            for i, hyp in enumerate(hypotheses, 1):
                report += f"{t['statement']} {i} ({t['score']}: {hyp.get('score', 0):.2f})\n"
                report += f"  {t['statement']}: {hyp.get('statement', 'N/A')}\n"
                report += f"  {t['mechanism']}: {hyp.get('mechanism', 'N/A')}\n"
                report += f"  {t['novelty']}: {hyp.get('novelty', 0):.2f}\n"
                report += f"  {t['impact']}: {hyp.get('impact', 0):.2f}\n"
                report += f"  {t['risk']}: {hyp.get('risk', 0):.2f}\n"
                sources = hyp.get('sources', [])
                if sources:
                    report += f"  {t['sources_label']}: {', '.join(sources)}\n"
                report += f"  {t['explanation']}: {hyp.get('explanation', 'N/A')}\n"
                report += f"  {t['recommendation']}: {hyp.get('recommendation', 'N/A')}\n"
                report += f"  {t['time']}: {hyp.get('generation_time', 'N/A')} сек\n\n"
            st.download_button(label="Скачать TXT", data=report, file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")