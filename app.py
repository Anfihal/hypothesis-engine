import streamlit as st
import json
import os
import sys
import tempfile
import re
from datetime import datetime

# === ЛОГИРОВАНИЕ ===
print("🚀 Загрузка приложения...")

# === БЕЗОПАСНЫЙ ИМПОРТ МОДУЛЕЙ ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.nlp.extractor import EntityExtractor
    print("✅ src.nlp.extractor")
except Exception as e:
    print(f"❌ Ошибка импорта EntityExtractor: {e}")
    st.error(f"Ошибка импорта: {e}")
    st.stop()

try:
    from src.graph.kgraph import KnowledgeGraph
    print("✅ src.graph.kgraph")
except Exception as e:
    print(f"❌ Ошибка импорта KnowledgeGraph: {e}")
    st.error(f"Ошибка импорта: {e}")
    st.stop()

try:
    from src.generation.deliberation import HypothesisDeliberation
    print("✅ src.generation.deliberation")
except Exception as e:
    print(f"❌ Ошибка импорта HypothesisDeliberation: {e}")
    st.error(f"Ошибка импорта: {e}")
    st.stop()

try:
    from src.ranking.scorer import rank_hypotheses
    print("✅ src.ranking.scorer")
except Exception as e:
    print(f"❌ Ошибка импорта rank_hypotheses: {e}")
    st.error(f"Ошибка импорта: {e}")
    st.stop()

# === ОПЦИОНАЛЬНЫЕ МОДУЛИ (PDF, OCR) ===
try:
    from PIL import Image
    import pytesseract
    print("✅ PIL + pytesseract")
except ImportError:
    pytesseract = None
    Image = None
    print("⚠️ PIL или pytesseract не установлены")

try:
    import PyPDF2
    print("✅ PyPDF2")
except ImportError:
    PyPDF2 = None
    print("⚠️ PyPDF2 не установлен")

try:
    from pdf2image import convert_from_path
    print("✅ pdf2image")
except ImportError:
    convert_from_path = None
    print("⚠️ pdf2image не установлен")

# === КЕШИРОВАННЫЕ ФУНКЦИИ ===
@st.cache_data(show_spinner=False)
def cached_extract_text_from_pdf(file_bytes):
    if PyPDF2 is None:
        return None
    try:
        from io import BytesIO
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text.strip() else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def cached_extract_text_from_image(file_bytes):
    if pytesseract is None or Image is None:
        return None
    try:
        from io import BytesIO
        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image, lang='eng+rus')
        return text.strip() if text.strip() else None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def cached_extract_text_from_pdf_ocr(file_bytes, poppler_path=None):
    if convert_from_path is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        images = convert_from_path(tmp_path, poppler_path=poppler_path, dpi=200)
        os.unlink(tmp_path)
        full_text = ""
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang='eng+rus')
            if text:
                full_text += f"--- Page {i+1} ---\n" + text + "\n"
        return full_text.strip() if full_text.strip() else None
    except Exception:
        return None

# === МУЛЬТИЯЗЫЧНЫЕ ТЕКСТЫ ===
TEXTS = {
    'en': {
        'title': "🧪 Scientific Hypothesis Generator",
        'params': "Parameters",
        'kpi': "Target KPI",
        'constraints': "Constraints",
        'model': "Model",
        'provider': "Provider",
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
        'empty_graph': "Graph is empty.",
        'not_specified': "not specified",
        'enable_search': "🔍 Enable internet search",
        'example': """The addition of 0.5% niobium to Inconel 718 significantly increases yield strength at elevated temperatures.
Previous studies show that niobium forms stable carbides which hinder dislocation motion.
However, excess niobium may cause embrittlement due to phase precipitation.
Recent experiments indicate that a two-step aging treatment can further enhance creep resistance.
Also, the use of tantalum has been explored but is expensive and limited in supply.
Grain boundary engineering through thermomechanical processing improves durability."""
    },
    'ru': {
        'title': "🧪 Генератор научных гипотез",
        'params': "Параметры",
        'kpi': "Целевой KPI",
        'constraints': "Ограничения",
        'model': "Модель",
        'provider': "Провайдер",
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
        'empty_graph': "Граф пуст.",
        'not_specified': "не указаны",
        'enable_search': "🔍 Включить поиск в интернете",
        'example': """Добавление 0.5% ниобия в сплав Inconel 718 значительно повышает предел текучести при повышенных температурах.
Предыдущие исследования показывают, что ниобий образует стабильные карбиды, которые препятствуют движению дислокаций.
Однако избыток ниобия может вызвать охрупчивание из-за выделения фаз.
Недавние эксперименты показывают, что двухступенчатая обработка старением может дополнительно повысить сопротивление ползучести.
Также изучалось использование тантала, но он дорог и ограничен в поставках.
Улучшение границ зёрен с помощью термомеханической обработки повышает долговечность."""
    },
    'zh': {
        'title': "🧪 科学假设生成器",
        'params': "参数",
        'kpi': "目标 KPI",
        'constraints': "约束条件",
        'model': "模型",
        'provider': "提供商",
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
        'empty_graph': "图谱为空。",
        'not_specified': "未指定",
        'enable_search': "🔍 启用互联网搜索",
        'example': """在Inconel 718中添加0.5%的铌可显著提高高温下的屈服强度。
先前的研究表明，铌形成稳定的碳化物，阻碍位错运动。
然而，过量的铌可能因相析出而导致脆化。
最近的实验表明，两步时效处理可以进一步提高抗蠕变性。
此外，还探索了钽的使用，但钽价格昂贵且供应有限。
通过热机械加工进行晶界工程可提高耐久性。"""
    }
}

# === ИНИЦИАЛИЗАЦИЯ SESSION_STATE ===
if 'lang' not in st.session_state:
    st.session_state.lang = 'ru'
if 'prev_lang' not in st.session_state:
    st.session_state.prev_lang = st.session_state.lang
if 'kpi' not in st.session_state:
    st.session_state.kpi = "Increase creep resistance of nickel-based superalloy by 20% at 700°C"
if 'constraints' not in st.session_state:
    st.session_state.constraints = "No rhenium, budget < $500k, scalable to industrial production"
if 'literature' not in st.session_state:
    st.session_state.literature = TEXTS[st.session_state.lang]['example']
if 'literature_edited' not in st.session_state:
    st.session_state.literature_edited = False
if 'provider' not in st.session_state:
    st.session_state.provider = "ollama"
if 'model_name' not in st.session_state:
    st.session_state.model_name = "llama3.1:8b"
if 'enable_search' not in st.session_state:
    st.session_state.enable_search = False
if 'hypotheses' not in st.session_state:
    st.session_state.hypotheses = None
if 'kg' not in st.session_state:
    st.session_state.kg = None

print("✅ session_state инициализирован")

# === НАСТРОЙКА СТРАНИЦЫ ===
st.set_page_config(
    page_title="Генератор гипотез" if st.session_state.lang == 'ru' else "Hypothesis Generator" if st.session_state.lang == 'en' else "假设生成器",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === ОТОБРАЖЕНИЕ ===
lang = st.session_state.lang
t = TEXTS[lang]

if lang != st.session_state.prev_lang and not st.session_state.literature_edited:
    st.session_state.literature = t['example']
st.session_state.prev_lang = lang

st.title(t['title'])
st.markdown("---")

# --- Боковая панель ---
with st.sidebar:
    new_lang = st.selectbox(
        t['lang'],
        options=["ru", "en", "zh"],
        index=["ru", "en", "zh"].index(lang),
        key="lang_selector"
    )
    if new_lang != lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.header(t['params'])
    st.session_state.kpi = st.text_area(t['kpi'], value=st.session_state.kpi, key="kpi_input")
    st.session_state.constraints = st.text_area(t['constraints'], value=st.session_state.constraints, key="constraints_input")

    st.subheader(t['model'])
    provider = st.selectbox(
        t['provider'],
        options=["ollama", "g4f", "yandex"],
        index=["ollama", "g4f", "yandex"].index(st.session_state.provider),
        key="provider_selector"
    )
    st.session_state.provider = provider

    if provider == "ollama":
        st.session_state.model_name = st.selectbox(
            t['local_model'],
            ["llama3.1:8b", "mistral:7b", "llama3.2:3b"],
            index=0,
            key="ollama_model"
        )
    elif provider == "g4f":
        st.session_state.model_name = st.selectbox(
            "G4F модель",
            ["gpt-4o-mini", "gpt-4o", "claude-3-haiku", "gemini-1.5-flash"],
            index=0,
            key="g4f_model"
        )
    else:
        st.session_state.model_name = "yandexgpt"

    st.session_state.enable_search = st.checkbox(
        t['enable_search'],
        value=st.session_state.enable_search,
        key="enable_search_check"
    )
    st.caption("По умолчанию локальная модель. Для g4f нужен интернет, для yandex – ключи в .env. Поиск через DuckDuckGo (без ключей).")

# --- Основная область ---
st.header(t['sources'])

def on_text_change():
    st.session_state.literature_edited = True

st.session_state.literature = st.text_area(
    t['text_input'],
    value=st.session_state.literature,
    height=200,
    key="literature_input",
    on_change=on_text_change
)

uploaded_files = st.file_uploader(
    t['upload'],
    type=["txt", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files:
    for file in uploaded_files:
        ext = file.name.split('.')[-1].lower()
        file_bytes = file.read()
        if ext == "txt":
            try:
                content = file_bytes.decode("utf-8")
                st.session_state.literature += "\n\n" + content
                st.session_state.literature_edited = True
                st.success(f"Добавлен текст из {file.name}")
            except Exception as e:
                st.error(f"Ошибка чтения {file.name}: {e}")
        elif ext == "pdf":
            text_from_pdf = cached_extract_text_from_pdf(file_bytes)
            if text_from_pdf and len(text_from_pdf) > 100:
                st.session_state.literature += "\n\n" + text_from_pdf
                st.session_state.literature_edited = True
                st.success(f"Извлечён текст из PDF {file.name}")
            else:
                ocr_text = cached_extract_text_from_pdf_ocr(file_bytes, poppler_path=None)
                if ocr_text:
                    st.session_state.literature += "\n\n" + ocr_text
                    st.session_state.literature_edited = True
                    st.success(f"Извлечён текст через OCR из {file.name}")
                else:
                    st.warning(f"Не удалось извлечь текст из {file.name}")
        elif ext in ["png", "jpg", "jpeg"]:
            ocr_text = cached_extract_text_from_image(file_bytes)
            if ocr_text:
                st.session_state.literature += "\n\n" + ocr_text
                st.session_state.literature_edited = True
                st.success(f"Распознан текст из изображения {file.name}")
            else:
                st.warning(f"Не удалось распознать текст из {file.name}")

# --- Кнопка генерации ---
if st.button(t['generate'], type="primary"):
    if not st.session_state.literature.strip():
        st.error(t['error_no_text'])
    else:
        with st.spinner("Идёт генерация гипотез... (может занять 1–3 минуты)"):
            try:
                extractor = EntityExtractor()
                analysis = extractor.process_document(st.session_state.literature)
                kg = KnowledgeGraph()
                for ent in analysis['entities']:
                    kg.add_entity(ent['text'], ent['label'])
                for rel in analysis['relationships']:
                    kg.add_relation(rel['subject'], rel['relation'], rel['object'], rel['evidence'])
                graph_context = kg.to_text_context()

                delib = HypothesisDeliberation(
                    provider=st.session_state.provider,
                    model=st.session_state.model_name,
                    timeout=600,
                    enable_search=st.session_state.enable_search
                )

                full_context = graph_context + "\nLiterature insights: " + "; ".join([
                    f"{rel['subject']} {rel['relation']} {rel['object']}"
                    for rel in analysis['relationships'][:5]
                ])

                hypotheses = delib.run(
                    st.session_state.kpi,
                    st.session_state.constraints,
                    full_context,
                    language=lang,
                    max_rounds=2
                )

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
                st.session_state.hypotheses = ranked
                st.session_state.kg = kg
                st.session_state.last_search_results = delib.last_search_results if hasattr(delib, 'last_search_results') else []
                st.success(t['success'].format(count=len(ranked)))

            except Exception as e:
                st.error(t['error_general'].format(error=str(e)))
                st.exception(e)

# --- Отображение результатов ---
if st.session_state.hypotheses is not None:
    st.markdown("---")
    st.header(t['results'])

    if st.session_state.kg is not None:
        with st.expander(t['graph'], expanded=False):
            kg = st.session_state.kg
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

    # Отображение найденных источников (если есть)
    if hasattr(st.session_state, 'last_search_results') and st.session_state.last_search_results:
        st.subheader("🌐 Найденные источники")
        for res in st.session_state.last_search_results[:8]:
            with st.expander(f"📄 {res.get('title', 'Без названия')}"):
                st.write(f"**Источник:** {res.get('source', '')}")
                st.write(f"**Проверенность:** {res.get('verification', '')}")
                st.write(f"**Аннотация:** {res.get('abstract', '')[:500]}...")
                if res.get('url'):
                    st.markdown(f"**Ссылка:** [{res['url']}]({res['url']})")

    # Гипотезы
    hypotheses = st.session_state.hypotheses
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
                # Отображаем источники с возможными ссылками
                sources_display = []
                for src in sources:
                    if src.startswith('http'):
                        sources_display.append(f"[{src}]({src})")
                    else:
                        sources_display.append(src)
                st.markdown(f"**{t['sources_label']}:** {', '.join(sources_display)}")
            else:
                st.markdown(f"**{t['sources_label']}:** {t['not_specified']}")
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
                "kpi": st.session_state.kpi,
                "constraints": st.session_state.constraints,
                "language": lang,
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
        if st.button(t['export_txt']):
            report = f"=== {t['title']} ===\n"
            report += f"{t['kpi']}: {st.session_state.kpi}\n"
            report += f"{t['constraints']}: {st.session_state.constraints}\n"
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
            st.download_button(
                label="Скачать TXT",
                data=report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

print("✅ Приложение успешно загружено")