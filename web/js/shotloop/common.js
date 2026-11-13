// Shot Planner: constants and small helpers shared by the panel and its views (no side effects: ComfyUI loads every
// .js under web/js as an extension, so this module only exports).
import { h as vueH, reactive, ref } from "../vendor/vue.esm-browser.prod.mjs";

// UI language is a browser preference, not part of the workflow or generation prompts.
const LANG_KEY = "bfs.shotloop.language";
const savedLanguage = (() => { try { return localStorage.getItem(LANG_KEY) || "en"; } catch { return "en"; } })();
export const uiLanguage = ref(["en", "pt", "es", "zh"].includes(savedLanguage) ? savedLanguage : "en");
export function setUiLanguage(value) {
  if (!["en", "pt", "es", "zh"].includes(value)) return;
  uiLanguage.value = value;
  try { localStorage.setItem(LANG_KEY, value); } catch { /* private mode */ }
}

const TEXT = {
  "🎞 Video": ["🎞 Vídeo", "🎞 Vídeo", "🎞 视频"],
  "🎬 Shots": ["🎬 Shots", "🎬 Planos", "🎬 镜头"],
  "▶ Run": ["▶ Executar", "▶ Ejecutar", "▶ 运行"],
  "📖 Guide": ["📖 Guia", "📖 Guía", "📖 指南"],
  "🧭 Tour": ["🧭 Tour", "🧭 Recorrido", "🧭 引导"],
  "Language": ["Idioma", "Idioma", "语言"],
  "Source video": ["Vídeo de origem", "Vídeo de origen", "源视频"],
  "Range": ["Trecho", "Rango", "范围"],
  "Timeline": ["Linha do tempo", "Línea de tiempo", "时间轴"],
  "zoom": ["zoom", "zoom", "缩放"],
  "Click a shot to select it (← → move between shots) · drag the white handles to move a boundary · double-click the shot bar to split · Delete removes the selected shot's cut": ["Clique em um plano para selecioná-lo (← → navega entre planos) · arraste as alças brancas para mover uma divisão · dê dois cliques na barra para dividir · Delete remove o corte do plano selecionado", "Haz clic en un plano para seleccionarlo (← → cambia de plano) · arrastra los controles blancos para mover un límite · haz doble clic en la barra para dividir · Delete elimina el corte del plano seleccionado", "点击镜头以选中（← → 切换镜头）· 拖动白色控点移动分界 · 双击镜头条进行拆分 · 按 Delete 删除所选镜头的切点"],
  "camera cuts or fixed length, on the model's frame grid": ["cortes de câmera ou duração fixa, na grade de frames do modelo", "cortes de cámara o duración fija, según la cuadrícula de fotogramas del modelo", "按镜头切换或固定时长，并符合模型帧步长"],
  "Range length, seconds (0 = to the end)": ["Duração do trecho, em segundos (0 = até o fim)", "Duración del tramo, en segundos (0 = hasta el final)", "片段时长（秒，0 = 到结尾）"],
  "refresh the input folder": ["atualizar a pasta de entrada", "actualizar la carpeta de entrada", "刷新输入文件夹"],
  "Pick or upload a video to start": ["Escolha ou envie um vídeo para começar", "Elige o sube un vídeo para empezar", "选择或上传视频以开始"],
  "The outputs carry a silent track of the right length, so Create Video works": ["As saídas incluem uma faixa silenciosa com a duração correta para funcionar em Create Video", "Las salidas incluyen una pista silenciosa con la duración correcta para que funcione Create Video", "输出会带有时长正确的静音音轨，以便 Create Video 正常工作"],
  "Audio of the planner's and the join's outputs": ["Áudio das saídas do Planner e do Join", "Audio de las salidas del Planner y del Join", "Planner 和 Join 输出的音频"],
  "the range ends after the player's frame": ["o trecho termina após o frame atual do player", "el rango termina después del fotograma actual", "范围在播放器当前帧之后结束"],
  "Play or scrub the video (or click the timeline), then ⇤ / ⇥. {…} frames ({…} s) of {…}: only this part is split, analysed, generated and joined.": ["Reproduza ou avance o vídeo (ou clique na linha do tempo) e use ⇤ / ⇥. {…} frames ({…} s) de {…}: somente este trecho será dividido, analisado, gerado e unido.", "Reproduce o desplaza el vídeo (o haz clic en la línea de tiempo) y usa ⇤ / ⇥. {…} fotogramas ({…} s) de {…}: solo se dividirá, analizará, generará y unirá este tramo.", "播放或拖动视频（也可点击时间轴），然后按 ⇤ / ⇥。共 {…}/{…} 帧（{…} 秒）：仅拆分、分析、生成并合并此片段。"],
  "15 s in one shot": ["15 s em um único plano", "15 s en un solo plano", "单镜头 15 秒"],
  "from the start frame · {…} frames": ["a partir do frame inicial · {…} frames", "desde el fotograma inicial · {…} fotogramas", "从起始帧开始 · {…} 帧"],
  "before the start: not planned": ["antes do início: fora do plano", "antes del inicio: no incluido", "起点之前：不处理"],
  "after the end: not planned": ["depois do fim: fora do plano", "después del final: no incluido", "终点之后：不处理"],
  "Split into shots": ["Dividir em planos", "Dividir en planos", "拆分镜头"],
  "Generation size": ["Tamanho da geração", "Tamaño de generación", "生成尺寸"],
  "Camera cuts": ["Cortes de câmera", "Cortes de cámara", "镜头切换"],
  "Fixed length": ["Duração fixa", "Duración fija", "固定时长"],
  "One shot (whole range)": ["Um plano (trecho inteiro)", "Un plano (rango completo)", "单个镜头（整个范围）"],
  "Mode": ["Modo", "Modo", "模式"],
  "Detector": ["Detector", "Detector", "检测器"],
  "Sensitivity": ["Sensibilidade", "Sensibilidad", "灵敏度"],
  "Frame grid": ["Grade de frames", "Múltiplo de fotogramas", "帧步长"],
  "Max seconds / shot": ["Máx. segundos por plano", "Máx. segundos por plano", "每个镜头最长秒数"],
  "Min seconds / shot": ["Mín. segundos por plano", "Mín. segundos por plano", "每个镜头最短秒数"],
  "Timeline fps": ["FPS da linha do tempo", "FPS de la línea de tiempo", "时间轴帧率"],
  "Megapixels": ["Megapixels", "Megapíxeles", "百万像素"],
  "Size multiple": ["Múltiplo do tamanho", "Múltiplo del tamaño", "尺寸倍数"],
  "Size mode": ["Modo de tamanho", "Modo de tamaño", "尺寸模式"],
  "Megapixel target": ["Alvo em megapixels", "Objetivo en megapíxeles", "百万像素目标"],
  "Original video": ["Vídeo original", "Vídeo original", "原视频"],
  "Custom dimensions": ["Dimensões personalizadas", "Dimensiones personalizadas", "自定义尺寸"],
  "Output size": ["Tamanho de saída", "Tamaño de salida", "输出尺寸"],
  "Custom width": ["Largura personalizada", "Ancho personalizado", "自定义宽度"],
  "Custom height": ["Altura personalizada", "Alto personalizado", "自定义高度"],
  "Image max side (VLM)": ["Lado máximo da imagem (VLM)", "Lado máximo de imagen (VLM)", "VLM 图像最长边"],
  "VLM image sizing": ["Redimensiona as imagens enviadas ao VLM, mantendo a proporção. O processador do modelo ainda pode redimensioná-las.", "Ajusta las imágenes enviadas al VLM manteniendo la proporción. El procesador del modelo puede volver a ajustarlas.", "调整发送给 VLM 的图像并保持比例。模型处理器仍可能再次缩放。"],
  "Original resolution is rounded to the size multiple for model compatibility.": ["A resolução original é arredondada ao múltiplo definido para compatibilidade com o modelo.", "La resolución original se redondea al múltiplo indicado para compatibilidad con el modelo.", "原始分辨率会取整到尺寸倍数，以兼容模型。"],
  "✂ Auto split": ["✂ Dividir automaticamente", "✂ División automática", "✂ 自动拆分"],
  "↻ Re-analyse": ["↻ Analisar novamente", "↻ Volver a analizar", "↻ 重新分析"],
  "Continuity": ["Continuidade", "Continuidad", "连续性"],
  "Copy to other shots": ["Copiar para outros planos", "Copiar a otros planos", "复制到其他镜头"],
  "Describe references ({details})": ["Descrever referências ({details})", "Describir referencias ({details})", "描述参考图（{details}）"],
  "VLM": ["VLM", "VLM", "视觉语言模型"],
  "Use the VLM when the workflow runs": ["Usar o VLM ao executar o workflow", "Usar el VLM al ejecutar el flujo", "运行工作流时使用 VLM"],
  "Fill {shot} in the prompt": ["Preencher {shot} no prompt", "Completar {shot} en el prompt", "在提示词中填入 {shot}"],
  "Frames per shot": ["Frames por plano", "Fotogramas por plano", "每个镜头的帧数"],
  "Max tokens": ["Máx. tokens", "Máx. tokens", "最大 token 数"],
  "🤖 Analyse shots": ["🤖 Analisar planos", "🤖 Analizar planos", "🤖 分析镜头"],
  "📝 Describe refs": ["📝 Descrever referências", "📝 Describir referencias", "📝 描述参考图"],
  "🎨 Colored Mark": ["🎨 Marcação colorida", "🎨 Marca de color", "🎨 彩色标记"],
  "Colored Mark": ["Marca colorida", "Marca de color", "彩色标记"],
  "👁 Preview": ["👁 Prévia", "👁 Vista previa", "👁 预览"],
  "Full frame": ["Quadro inteiro", "Fotograma entero", "整个画面"],
  "Red": ["Vermelho", "Rojo", "红色"],
  "Checks": ["Verificações", "Comprobaciones", "检查"],
  "Run": ["Executar", "Ejecutar", "运行"],
  "Queue loop (one shot per run)": ["Fila (um plano por execução)", "Cola (un plano por ejecución)", "队列循环（每次运行一个镜头）"],
  "Skipped shots in the output": ["Planos ignorados na saída", "Planos omitidos en la salida", "输出中跳过的镜头"],
  "Keep original video": ["Manter o vídeo original", "Conservar el vídeo original", "保留原视频"],
  "Remove them": ["Removê-los", "Quitarlos", "移除"],
  "Max shots (0 = all)": ["Máx. planos (0 = todos)", "Máx. planos (0 = todos)", "镜头上限（0 = 全部）"],
  "Auto-queue the next shot": ["Enfileirar próximo plano automaticamente", "Añadir automáticamente el siguiente plano a la cola", "自动将下一个镜头加入队列"],
  "Video": ["Vídeo", "Vídeo", "视频"],
  "Shots": ["Planos", "Planos", "镜头"],
  "Prompts": ["Prompts", "Prompts", "提示词"],
  "Back": ["Voltar", "Atrás", "上一步"],
  "Next": ["Avançar", "Siguiente", "下一步"],
  "Close tour": ["Fechar tour", "Cerrar recorrido", "关闭引导"],
  "Shot Planner tour": ["Tour do Shot Planner", "Recorrido de Shot Planner", "Shot Planner 引导"],
  "Shot": ["Plano", "Plano", "镜头"],
  "This tour is optional. Use Next to visit each main tab; close it whenever you like.": ["Este tour é opcional. Use Avançar para conhecer cada aba; feche quando quiser.", "Este recorrido es opcional. Pulsa Siguiente para visitar cada pestaña; puedes cerrarlo cuando quieras.", "此引导为可选。点击下一步浏览各个主选项卡，也可随时关闭。"],
  "🎬 Shot Planner": ["🎬 Planejador de planos", "🎬 Planificador de planos", "🎬 镜头规划器"],
  "— choose a video from the input folder —": ["— escolha um vídeo da pasta de entrada —", "— elige un vídeo de la carpeta de entrada —", "— 从输入文件夹选择视频 —"],
  "⬆ Upload": ["⬆ Enviar", "⬆ Subir", "⬆ 上传"],
  "It is split into shots the model can generate; every shot then gets its own reference, prompt and mask.": ["O vídeo será dividido em planos que o modelo consegue gerar; cada plano terá sua própria referência, prompt e máscara.", "El vídeo se divide en planos que el modelo puede generar; cada uno tendrá su referencia, prompt y máscara.", "视频会拆分为模型可生成的镜头，每个镜头都有自己的参考图、提示词和蒙版。"],
  "work on a part of the video": ["trabalhe em um trecho do vídeo", "trabaja en una parte del vídeo", "处理视频的一部分"],
  "Start frame": ["Frame inicial", "Fotograma inicial", "起始帧"],
  "End frame (0 = end)": ["Frame final (0 = fim)", "Fotograma final (0 = final)", "结束帧（0 = 末尾）"],
  "⇤ start here": ["⇤ começar aqui", "⇤ empezar aquí", "⇤ 从此处开始"],
  "⇥ end here": ["⇥ terminar aqui", "⇥ terminar aquí", "⇥ 在此处结束"],
  "✕ whole video": ["✕ vídeo inteiro", "✕ vídeo completo", "✕ 整个视频"],
  "re-analyses": ["reanalisar", "vuelve a analizar", "重新分析"],
  "ignores Max seconds / shot": ["ignora Máx. segundos por plano", "ignora Máx. segundos por plano", "忽略每个镜头的最长秒数"],
  "cropped shots use it for the crop": ["planos recortados usam esse valor no recorte", "los planos recortados usan este valor", "裁剪镜头会使用此尺寸"],
  "🎞 Mask video (whole video)": ["🎞 Vídeo de máscara (vídeo inteiro)", "🎞 Vídeo de máscara (todo el vídeo)", "🎞 蒙版视频（整段视频）"],
  "Fill holes": ["Preencher buracos", "Rellenar huecos", "填充孔洞"],
  "Invert": ["Inverter", "Invertir", "反转"],
  "Expand (px)": ["Expandir (px)", "Expandir (px)", "扩展（像素）"],
  "Temporal expand (frames)": ["Expansão temporal (frames)", "Expansión temporal (fotogramas)", "时间扩展（帧）"],
  "Detection threshold": ["Limiar de detecção", "Umbral de detección", "检测阈值"],
  "Max objects": ["Máx. objetos", "Máx. objetos", "最大对象数"],
  "Crop padding (%)": ["Margem do recorte (%)", "Margen del recorte (%)", "裁剪边距（%）"],
  "Feather (px)": ["Suavização da borda (px)", "Suavizado del borde (px)", "羽化（像素）"],
  "Paste back": ["Colar de volta", "Pegar de vuelta", "贴回方式"],
  "only the mask (feathered)": ["somente a máscara (suavizada)", "solo la máscara (suavizada)", "仅蒙版（羽化边缘）"],
  "the whole box (feathered)": ["a caixa inteira (suavizada)", "todo el cuadro (suavizado)", "整个框（羽化边缘）"],
  "Blockify (px, 0 = off)": ["Blocos (px, 0 = desligado)", "Bloques (px, 0 = desactivado)", "块化（像素，0 = 关闭）"],
  "SAM 3": ["SAM 3", "SAM 3", "SAM 3"],
  "✕ clear": ["✕ limpar", "✕ borrar", "✕ 清除"],
  "✕ clear all": ["✕ limpar tudo", "✕ borrar todo", "✕ 全部清除"],
  "Use global": ["Usar global", "Usar global", "使用全局设置"],
  "🎯 Points…": ["🎯 Pontos…", "🎯 Puntos…", "🎯 点…"],
  "🎞 mask video": ["🎞 vídeo de máscara", "🎞 vídeo de máscara", "🎞 蒙版视频"],
  "↳ global prompt": ["↳ prompt global", "↳ prompt global", "↳ 全局提示词"],
  "— no prompt —": ["— sem prompt —", "— sin prompt —", "— 无提示词 —"],
  "▶ Play all": ["▶ Reproduzir todos", "▶ Reproducir todos", "▶ 播放全部"],
  "■ Stop": ["■ Parar", "■ Detener", "■ 停止"],
  "loop shot": ["repetir plano", "repetir plano", "循环播放镜头"],
  "Click the timeline to seek. Play all skips shots that will not run.": ["Clique na linha do tempo para buscar. Reproduzir todos ignora planos que não serão executados.", "Haz clic en la línea de tiempo para buscar. Reproducir todos omite los planos que no se ejecutarán.", "点击时间轴进行定位。播放全部会跳过不运行的镜头。"],
  "Custom": ["Personalizado", "Personalizado", "自定义"],
  "short": ["curto", "breve", "简短"],
  "full body": ["corpo inteiro", "cuerpo entero", "全身"],
  "head / face": ["cabeça / rosto", "cabeza / rostro", "头部/面部"],
  "face attributes": ["características do rosto", "rasgos faciales", "面部特征"],
  "outfit": ["roupa", "ropa", "服装"],
  "Apply suggestions → all": ["Aplicar sugestões → todos", "Aplicar sugerencias → todos", "应用建议 → 全部"],
  "fit the prompt": ["ajustar ao prompt", "ajustar al prompt", "适配提示词"],
  "🎯 Track → whole video": ["🎯 Rastrear → vídeo inteiro", "🎯 Seguir → vídeo completo", "🎯 跟踪 → 整个视频"],
  "Save (this shot)": ["Salvar (este plano)", "Guardar (este plano)", "保存（此镜头）"],
  "Save → all shots": ["Salvar → todos os planos", "Guardar → todos los planos", "保存 → 所有镜头"],
  "Cancel": ["Cancelar", "Cancelar", "取消"],
  "Clear points": ["Limpar pontos", "Borrar puntos", "清除点"],
  "👁 Segment": ["👁 Segmentar", "👁 Segmentar", "👁 分割"],
  "Apply these points to EVERY shot, each on its frame at the same relative position (a subject that stays in place)": ["Aplicar estes pontos a TODOS os planos, na mesma posição relativa (para um sujeito que não muda de lugar)", "Aplicar estos puntos a TODOS los planos, en la misma posición relativa (si el sujeto no se mueve)", "将这些点应用到每个镜头的相同相对位置（适用于位置不变的主体）"],
  "the range starts at the player's frame": ["o trecho começa no frame atual do player", "el rango empieza en el fotograma actual", "范围从播放器当前帧开始"],
  "Nothing to fix in the shots that run.": ["Nenhum problema nos planos que serão executados.", "No hay nada que corregir en los planos que se ejecutarán.", "将运行的镜头没有需要修复的问题。"],
  "all good": ["tudo certo", "todo bien", "一切正常"],
  "warning(s)": ["aviso(s)", "aviso(s)", "条警告"],
  "note(s)": ["observação(ões)", "nota(s)", "条备注"],
  "quick tests": ["testes rápidos", "pruebas rápidas", "快速测试"],
  "Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video.": ["Cada execução gera e salva um plano. Os nodes após BFS Shot Join só executam no último, com o vídeo completo.", "Cada ejecución genera y guarda un plano. Los nodos posteriores a BFS Shot Join solo se ejecutan en el último, con el vídeo completo.", "每次运行生成并保存一个镜头。BFS Shot Join 后的节点仅在最后一个镜头运行，并接收完整视频。"],
  "only shots that run": ["somente planos executados", "solo los planos que se ejecutan", "仅统计将运行的镜头"],
  "off": ["desligado", "desactivado", "关闭"],
  "on": ["ativado", "activado", "开启"],
  "run": ["executar", "ejecutar", "运行"],
  "skip": ["ignorar", "omitir", "跳过"],
  "⏸ Disable": ["⏸ Desativar", "⏸ Desactivar", "⏸ 禁用"],
  "▶ Enable": ["▶ Ativar", "▶ Activar", "▶ 启用"],
  "✂ Split": ["✂ Dividir", "✂ Dividir", "✂ 拆分"],
  "⇥ Merge": ["⇥ Unir", "⇥ Unir", "⇥ 合并"],
  "Advanced": ["Avançado", "Avanzado", "高级"],
  "English": ["Inglês", "Inglés", "英语"],
  "Português": ["Português", "Portugués", "葡萄牙语"],
  "Español": ["Espanhol", "Español", "西班牙语"],
  "中文": ["Chinês", "Chino", "中文"],
  "Audio": ["Áudio", "Audio", "音频"],
  "use the video's audio": ["usar o áudio do vídeo", "usar el audio del vídeo", "使用视频音频"],
  "silent track": ["faixa silenciosa", "pista silenciosa", "静音音轨"],
  "auto": ["automático", "automático", "自动"],
  "fixed": ["fixo", "fijo", "固定"],
  "single": ["único", "único", "单个"],
  "adaptive": ["adaptativo", "adaptativo", "自适应"],
  "content": ["conteúdo", "contenido", "内容"],
  "builtin": ["integrado", "integrado", "内置"],
  "PySceneDetect adaptive": ["PySceneDetect adaptativo", "PySceneDetect adaptativo", "PySceneDetect 自适应"],
  "PySceneDetect content": ["PySceneDetect por conteúdo", "PySceneDetect por contenido", "PySceneDetect 内容检测"],
  "Built-in": ["Integrado", "Integrado", "内置"],
  "Auto loop": ["Loop automático", "Bucle automático", "自动循环"],
  "Queue loop": ["Loop em fila", "Bucle en cola", "队列循环"],
  "queue loop": ["loop em fila", "bucle en cola", "队列循环"],
  "auto loop": ["loop automático", "bucle automático", "自动循环"],
  "no audio": ["sem áudio", "sin audio", "无音频"],
  "timeline": ["linha do tempo", "línea de tiempo", "时间轴"],
  "generate at": ["gerar em", "generar a", "生成尺寸"],
  "Describe target": ["Descrever alvo", "Describir objetivo", "描述目标"],
  "🧑 Describe target": ["🧑 Descrever alvo", "🧑 Describir objetivo", "🧑 描述目标"],
  "new subject": ["novo sujeito", "nuevo sujeto", "新对象"],
  "Prompt": ["Prompt", "Prompt", "提示词"],
  "reference": ["referência", "referencia", "参考图"],
  "No shots yet": ["Nenhum plano ainda", "Aún no hay planos", "尚无镜头"],
  "cut": ["corte", "corte", "切换"],
  "people": ["pessoas", "personas", "人物"],
  "person": ["pessoa", "persona", "人物"],
  "shots": ["planos", "planos", "镜头"],
  "shot(s)": ["plano(s)", "plano(s)", "个镜头"],
  "main": ["principal", "principal", "主要"],
  "Connect a Qwen3-VL": ["Conecte um Qwen3-VL", "Conecta un Qwen3-VL", "连接 Qwen3-VL"],
  "reset": ["redefinir", "restablecer", "重置"],
  "all shots": ["todos os planos", "todos los planos", "所有镜头"],
  "this shot": ["este plano", "este plano", "此镜头"],
  "Settings": ["Configurações", "Ajustes", "设置"],
  "mask text": ["texto da máscara", "texto de máscara", "蒙版描述"],
  "optional": ["opcional", "opcional", "可选"],
  "woman in pink top": ["mulher de blusa rosa", "mujer con blusa rosa", "穿粉色上衣的女人"],
  "nothing found": ["nada encontrado", "no se encontró nada", "未找到内容"],
  "Preview": ["Prévia", "Vista previa", "预览"],
  "per shot (planner)": ["por plano (planner)", "por plano (planificador)", "逐镜头（规划器）"],
  "shot": ["plano", "plano", "镜头"],
  "mask covers": ["máscara cobre", "la máscara cubre", "蒙版覆盖"],
  "Camera cuts: shots start at the detected cuts, long ones are split evenly. Fixed length: equal parts no longer than the maximum.": ["Cortes de câmera: os planos começam nos cortes detectados; planos longos são divididos igualmente. Duração fixa: partes iguais limitadas pelo máximo.", "Cortes de cámara: los planos empiezan en los cortes detectados; los largos se dividen por igual. Duración fija: partes iguales con el máximo indicado.", "镜头切换：镜头从检测到的切点开始，较长镜头会均匀拆分。固定时长：拆分为不超过上限的等长片段。"],
  "Cut detector. PySceneDetect adaptive works on most footage (fast camera moves don't trigger it); content is stricter on hard cuts; built-in needs no package.": ["Detector de cortes. O modo adaptativo do PySceneDetect funciona na maioria dos vídeos (movimentos rápidos de câmera não acionam cortes); o modo conteúdo é mais rigoroso; o integrado não exige pacote.", "Detector de cortes. El modo adaptativo de PySceneDetect funciona en la mayoría de vídeos (los movimientos rápidos no activan cortes); contenido es más estricto; el integrado no requiere paquetes.", "镜头切换检测器。PySceneDetect 自适应模式适用于大多数素材（快速运镜不会触发切点）；内容模式对硬切更严格；内置模式无需额外软件包。"],
  "How easily a change counts as a cut. Higher finds more cuts (and more false ones).": ["O quanto uma mudança é considerada corte. Um valor maior encontra mais cortes e também mais falsos positivos.", "Facilidad con la que un cambio cuenta como corte. Un valor mayor detecta más cortes y también más falsos positivos.", "变化被判定为切点的难易程度。数值越高，检测到的切点越多，误报也越多。"],
  "Frame counts the model accepts. H3: 17n+5, LTX/Wan: 8n+1, Wan: 4n+1. Each shot is generated at the next valid length and trimmed back.": ["Contagens de frames aceitas pelo modelo. H3: 17n+5; LTX/Wan: 8n+1; Wan: 4n+1. Cada plano é gerado no próximo comprimento válido e depois aparado.", "Cantidad de fotogramas que acepta el modelo. H3: 17n+5; LTX/Wan: 8n+1; Wan: 4n+1. Cada plano se genera con la siguiente duración válida y luego se recorta.", "模型接受的帧数规则。H3：17n+5；LTX/Wan：8n+1；Wan：4n+1。每个镜头按下一个有效长度生成后再裁短。"],
  "Longest shot sent to the model. Keep it within what the model / LoRA was trained on (H3 body swap: about 4.5 s).": ["Duração máxima enviada ao modelo. Mantenha dentro do que o modelo/LoRA aprendeu (body swap H3: cerca de 4,5 s).", "Duración máxima enviada al modelo. Respeta lo usado en el entrenamiento del modelo/LoRA (body swap H3: unos 4,5 s).", "发送给模型的镜头最长时长。应符合模型/LoRA 的训练长度（H3 换人约 4.5 秒）。"],
  "Shorter shots merge into a neighbour when the merge still fits the maximum.": ["Planos mais curtos são unidos ao vizinho quando a junção ainda respeita o máximo.", "Los planos más cortos se unen al vecino si la unión respeta el máximo.", "较短镜头会在合并后仍未超过上限时与相邻镜头合并。"],
  "The planner works on this frame rate (the source is resampled). Generated video and audio use it too.": ["O planner usa esta taxa de frames (o vídeo de origem é reamostrado). O vídeo e áudio gerados também a usam.", "El planificador usa esta frecuencia de fotogramas (el origen se remuestrea). El vídeo y el audio generados también la usan.", "规划器使用此帧率（源视频会重新采样）。生成的视频和音频也会使用此帧率。"],
  "Only the first N shots (0 = all). Handy for quick tests.": ["Somente os primeiros N planos (0 = todos). Útil para testes rápidos.", "Solo los primeros N planos (0 = todos). Útil para pruebas rápidas.", "仅处理前 N 个镜头（0 = 全部）。适合快速测试。"],
  "Width and height snap to this multiple (32 for H3 and most video models).": ["Largura e altura são ajustadas a este múltiplo (32 para H3 e a maioria dos modelos de vídeo).", "El ancho y alto se ajustan a este múltiplo (32 para H3 y la mayoría de modelos de vídeo).", "宽度和高度会对齐到此倍数（H3 和多数视频模型使用 32）。"],
  "Shots that do not run: keep the original video there, or remove them (with their audio).": ["Planos não executados: manter o vídeo original nesses trechos ou removê-los junto com o áudio.", "Planos que no se ejecutan: conserva el vídeo original en ese tramo o elimínalo junto con su audio.", "未运行的镜头：保留对应原视频片段，或连同音频一起移除。"],
  "Context kept around the mask's box, as a % of the box. More context blends better; less gives the model more pixels on the subject.": ["Contexto mantido ao redor da caixa da máscara, como porcentagem da caixa. Mais contexto mistura melhor; menos dá mais pixels ao sujeito.", "Contexto alrededor del cuadro de la máscara, como porcentaje. Más contexto mezcla mejor; menos ofrece más píxeles al sujeto.", "蒙版边框周围保留的上下文比例。上下文越多越容易融合；越少则主体像素越多。"],
  "Grow the mask by this many pixels: before pasting a cropped result back, and for the generation mask (inpaint 'only the mask'). Raise it when the new person is bigger than the old one (longer hair, wider body).": ["Expande a máscara por esta quantidade de pixels antes de colar o recorte e na máscara de geração. Aumente se a nova pessoa for maior (cabelo mais comprido, corpo mais largo).", "Amplía la máscara esta cantidad de píxeles antes de pegar el recorte y para la máscara de generación. Auméntala si la persona nueva es más grande.", "粘贴裁剪结果前以及生成蒙版时，将蒙版扩展指定像素。新人物更大时（头发更长、身体更宽）可调高。"],
  "Soft edge of the paste-back, in pixels.": ["Suavização da borda ao colar de volta, em pixels.", "Suavizado del borde al pegar de vuelta, en píxeles.", "贴回时边缘的柔化像素数。"],
  "Hold the mask this many frames before and after: removes flicker where SAM 3 drops a frame.": ["Estende a máscara por esta quantidade de frames antes e depois; reduz falhas quando o SAM 3 perde um frame.", "Mantiene la máscara estos fotogramas antes y después; reduce parpadeos cuando SAM 3 pierde uno.", "蒙版前后延续指定帧数，减少 SAM 3 漏掉某帧时的闪烁。"],
  "Square blocks of this size (0 = off). 16 matches H3's latent grid, useful as an inpainting mask.": ["Blocos quadrados deste tamanho (0 = desligado). 16 corresponde à grade latente do H3 e pode ser útil como máscara de inpainting.", "Bloques cuadrados de este tamaño (0 = desactivado). 16 coincide con la cuadrícula latente de H3 y sirve para inpainting.", "使用此尺寸的方块（0 = 关闭）。16 与 H3 潜空间网格匹配，可用于修补蒙版。"],
  "SAM 3 score needed to keep a text match. Lower finds more (and more wrong) objects.": ["Pontuação mínima do SAM 3 para manter uma correspondência por texto. Valores menores encontram mais objetos, inclusive incorretos.", "Puntuación mínima de SAM 3 para conservar una coincidencia de texto. Un valor menor encuentra más objetos, también erróneos.", "保留文本匹配所需的 SAM 3 分数。调低会找到更多对象，也会增加误检。"],
  "How many matches of the text prompt are tracked together.": ["Quantas correspondências do texto serão rastreadas juntas.", "Cuántas coincidencias del texto se siguen a la vez.", "同时跟踪多少个符合文本的对象。"],
  "Only the mask: the result replaces the masked pixels. The whole box: the full crop is pasted with feathered borders.": ["Somente máscara: o resultado substitui os pixels mascarados. Caixa inteira: todo o recorte é colado com bordas suaves.", "Solo máscara: el resultado sustituye los píxeles enmascarados. Cuadro completo: se pega todo el recorte con bordes suavizados.", "仅蒙版：结果替换蒙版像素。整个框：粘贴完整裁剪区域并柔化边缘。"],
  "How many frames of each shot the VLM looks at.": ["Quantos frames de cada plano o VLM analisa.", "Cuántos fotogramas de cada plano analiza el VLM.", "VLM 分析每个镜头的帧数。"],
  "Longest VLM answer. Raise it for long descriptions.": ["Tamanho máximo da resposta do VLM. Aumente para descrições longas.", "Longitud máxima de respuesta del VLM. Auméntala para descripciones largas.", "VLM 回复的最大长度。长描述可调高。"],
  "skipped by hand": ["ignorado manualmente", "omitido manualmente", "手动跳过"],
  "longer than {…} frames: split it": ["mais longo que {…} frames: divida o plano", "supera {…} fotogramas: divide el plano", "超过 {…} 帧：请拆分镜头"],
  "{…} needs a mask (text, points or a mask video; or the planner's mask input)": ["{…} precisa de uma máscara (texto, pontos, vídeo de máscara ou entrada de máscara do Planner)", "{…} necesita una máscara (texto, puntos, vídeo de máscara o entrada de máscara del planificador)", "{…} 需要蒙版（文本、点、蒙版视频或 Planner 的蒙版输入）"],
  "the mask preview found nothing": ["a prévia da máscara não encontrou nada", "la vista previa de la máscara no encontró nada", "蒙版预览未找到目标"],
  "the prompt uses {target} but the shot has no target description": ["o prompt usa {target}, mas o plano não tem uma descrição do alvo", "el prompt usa {target}, pero el plano no tiene una descripción del objetivo", "提示词使用了 {target}，但镜头没有目标描述"],
  "no prompt (fine when the node's prompt input is connected)": ["sem prompt (tudo certo se a entrada prompt do node estiver conectada)", "sin prompt (no pasa nada si la entrada prompt del nodo está conectada)", "没有提示词（若节点的 prompt 输入已连接则无需处理）"],
  "Analysing video…": ["Analisando o vídeo…", "Analizando el vídeo…", "正在分析视频…"],
  "Detecting cuts…": ["Detectando cortes…", "Detectando cortes…", "正在检测切点…"],
  "Selecting the person (SAM 3) and describing them (VLM)…": ["Selecionando a pessoa (SAM 3) e descrevendo-a (VLM)…", "Seleccionando a la persona (SAM 3) y describiéndola (VLM)…", "正在选择人物（SAM 3）并生成描述（VLM）…"],
  "The SAM 3 target cutout was created (shown below), but no VLM is loaded to describe it. Connect Qwen3-VL to the Shot Planner's vlm input and queue the workflow once; then click Describe target again. You can also type the target description yourself.": ["O recorte do alvo pelo SAM 3 foi criado abaixo, mas nenhum VLM está carregado para descrevê-lo. Conecte o Qwen3-VL à entrada vlm do Shot Planner e execute a fila uma vez; depois clique novamente em Descrever alvo. Você também pode escrever a descrição manualmente.", "SAM 3 creó el recorte del objetivo (se muestra abajo), pero no hay ningún VLM cargado para describirlo. Conecta Qwen3-VL a la entrada vlm de Shot Planner y ejecuta el flujo una vez; después vuelve a pulsar Describir objetivo. También puedes escribir la descripción.", "SAM 3 已生成目标抠图（见下方），但尚未加载 VLM 来描述它。将 Qwen3-VL 连接到 Shot Planner 的 vlm 输入并运行一次队列，然后再次点击“描述目标”。也可以手动填写描述。"],
  "Segmenting with SAM 3… (the first time downloads/loads the model)": ["Segmentando com SAM 3… (na primeira vez, o modelo será baixado/carregado)", "Segmentando con SAM 3… (la primera vez se descarga/carga el modelo)", "正在使用 SAM 3 分割…（首次运行会下载并加载模型）"],
  "open the checks (Run tab)": ["abrir as verificações (aba Executar)", "abrir las comprobaciones (pestaña Ejecutar)", "打开检查项（Run 标签页）"],
  "Open the full guide (every setting explained, with examples)": ["Abrir o guia completo (com explicações e exemplos de cada opção)", "Abrir la guía completa (explicaciones y ejemplos de cada opción)", "打开完整指南（包含各设置的说明和示例）"],
  "click to remove": ["clique para remover", "haz clic para quitar", "点击删除"],
  "text prompt (used when there are no points)": ["texto de seleção (usado quando não há pontos)", "texto de selección (se usa cuando no hay puntos)", "文本提示（没有标记点时使用）"],
  "Track this selection with SAM 3 through the whole video (or the range), forward and backward from this frame; every shot without its own selection takes its part of the track. Also type a short text (e.g. 'woman in pink top') so the subject is found again after camera cuts.": ["Rastreie esta seleção com SAM 3 pelo vídeo inteiro (ou trecho), avançando e retrocedendo a partir deste frame. Cada plano sem seleção própria usa sua parte do rastreamento. Digite também uma descrição curta (por exemplo, 'mulher de blusa rosa') para reencontrar o sujeito após os cortes.", "Sigue esta selección con SAM 3 por todo el vídeo (o tramo), hacia delante y atrás desde este fotograma. Cada plano sin selección propia usa su parte del seguimiento. Añade también una descripción breve (por ejemplo, 'mujer con blusa rosa') para volver a encontrar al sujeto después de los cortes.", "使用 SAM 3 从此帧开始向前和向后跟踪所选对象，覆盖整个视频（或片段）。没有单独选区的镜头会使用对应的跟踪结果。再输入简短描述（如“穿粉色上衣的女性”），以便在切镜后重新找到主体。"],
  "Fill enclosed gaps so each region is solid": ["Preencher espaços fechados para deixar cada região sólida", "Rellenar huecos cerrados para que cada región sea sólida", "填充封闭空隙，使区域完整"],
  "Overlay the mask preview on the shot cards": ["Sobrepor a prévia da máscara nos cartões dos planos", "Superponer la vista previa de la máscara en las tarjetas de los planos", "在镜头卡片上叠加蒙版预览"],
  "Global prompt (a connected `prompt` input overrides it)": ["Prompt global (a entrada `prompt` conectada tem prioridade)", "Prompt global (la entrada `prompt` conectada tiene prioridad)", "全局提示词（若连接了 `prompt` 输入，则以输入为准）"],
  "At run time the planner asks the VLM about the shots it runs and applies the options below": ["Durante a execução, o Planner consulta o VLM sobre os planos processados e aplica as opções abaixo", "Durante la ejecución, el planificador consulta al VLM sobre los planos y aplica las opciones siguientes", "运行时，Planner 会让 VLM 分析要处理的镜头并应用以下选项"],
  "Write {shot} in a prompt: it becomes the VLM's description of that shot (camera, framing, action)": ["Use {shot} no prompt para inserir a descrição do plano pelo VLM (câmera, enquadramento e ação)", "Escribe {shot} en el prompt para insertar la descripción del plano generada por el VLM (cámara, encuadre y acción)", "在提示词中写入 {shot}，即可插入 VLM 对镜头的描述（摄影机、构图和动作）"],
  "0 = always the same answer (greedy); 0.5-1 = varied answers, picked by the seed": ["0 = resposta sempre igual (greedy); 0,5–1 = respostas variadas, controladas pela seed", "0 = respuesta siempre igual (greedy); 0,5–1 = respuestas variadas según la seed", "0 = 始终生成相同答案（贪心）；0.5–1 = 由 seed 控制的多样答案"],
  "Used when the temperature is above 0: another seed, another description": ["Usada quando a temperatura é maior que 0: outra seed gera outra descrição", "Se usa cuando la temperatura es mayor que 0: otra seed genera otra descripción", "温度大于 0 时使用：更换 seed 可生成不同描述"],
  "✓ Nothing to fix in the shots that run.": ["✓ Nenhum ajuste necessário nos planos que serão executados.", "✓ No hay nada que corregir en los planos que se ejecutarán.", "✓ 将运行的镜头无需修正。"],
  "go to this shot": ["ir para este plano", "ir a este plano", "跳转到此镜头"],
  "Pick a video in the Video tab.": ["Escolha um vídeo na aba Vídeo.", "Elige un vídeo en la pestaña Vídeo.", "请在 Video 标签页中选择视频。"],
  "split this shot in two": ["dividir este plano em dois", "dividir este plano en dos", "将此镜头一分为二"],
  "merge with the next shot": ["unir ao próximo plano", "unir con el siguiente plano", "与下一个镜头合并"],
  "Forget the recent pictures (the ones this plan still uses stay)": ["Esquecer as imagens recentes (as usadas neste plano serão mantidas)", "Olvidar las imágenes recientes (se conservan las que usa este plan)", "清除最近使用的图片（本计划仍在使用的图片会保留）"],
  "Remove EVERY reference from this plan (global and all shots) and the recent list": ["Remover TODAS as referências deste plano (global e de todos os planos) e da lista recente", "Eliminar TODAS las referencias de este plan (globales y de todos los planos) y de la lista reciente", "从此计划（全局和所有镜头）及最近列表中移除所有参考图"],
  "nothing found: the shot runs without a mask": ["nada encontrado: o plano será executado sem máscara", "no se encontró nada: el plano se ejecutará sin máscara", "未找到目标：此镜头将不使用蒙版运行"],
  "previous shot as reference": ["plano anterior como referência", "plano anterior como referencia", "将上一镜头作为参考图"],
  "previous shot as first frame": ["plano anterior como primeiro frame", "plano anterior como primer fotograma", "将上一镜头作为首帧"],
  "previous shot's end as history (long take)": ["final do plano anterior como histórico (plano contínuo)", "final del plano anterior como historial (toma continua)", "将上一镜头的结尾作为历史（连续长镜头）"],
  "its first frame": ["seu primeiro frame", "su primer fotograma", "其首帧"],
  "its last frame": ["seu último frame", "su último fotograma", "其末帧"],
  "its middle frame": ["seu frame do meio", "su fotograma central", "其中间帧"],
  "✓ copied to {…} shot(s)": ["✓ copiado para {…} plano(s)", "✓ copiado a {…} plano(s)", "✓ 已复制到 {…} 个镜头"],
  "show mask previews": ["mostrar prévias das máscaras", "mostrar vistas previas de las máscaras", "显示蒙版预览"],
  "max total seconds": ["duração total máxima", "duración total máxima", "总时长上限"],
  "Remove every reference picture from this plan (global, every shot) and forget the recent list?": ["Remover todas as imagens de referência deste plano (global e de todos os planos) e limpar a lista recente?", "¿Eliminar todas las imágenes de referencia de este plan (globales y de todos los planos) y borrar la lista reciente?", "移除此计划中的所有参考图（全局及各镜头）并清空最近列表？"],
  "upload a mask video for this shot": ["enviar um vídeo de máscara para este plano", "subir un vídeo de máscara para este plano", "为此镜头上传蒙版视频"],
  "This shot uses its mask video: the text and points are ignored.": ["Este plano usa o vídeo de máscara; o texto e os pontos são ignorados.", "Este plano usa su vídeo de máscara; se ignoran el texto y los puntos.", "此镜头使用自己的蒙版视频，文本和标记点将被忽略。"],
  "copy (the {shot} placeholder in the prompt gets it automatically at run time)": ["copiar (o marcador {shot} do prompt recebe isso durante a execução)", "copiar (el marcador {shot} del prompt lo inserta automáticamente al ejecutar)", "复制（运行时会自动填入提示词中的 {shot}）"],
  "How much of the previous result's end goes before this shot. 5 frames is a short context; longer windows carry more motion and use more memory.": ["Quantos frames finais do resultado anterior são colocados antes deste plano. 5 frames dão pouco contexto; janelas maiores carregam mais movimento e usam mais memória.", "Cuántos fotogramas finales del resultado anterior se colocan antes de este plano. 5 fotogramas dan poco contexto; ventanas más largas conservan más movimiento y usan más memoria.", "将上一结果的多少末尾帧放在此镜头之前。5 帧是较短上下文；更长窗口保留更多动作，也占用更多内存。"],
  "Which frame of the previous shot's result": ["Qual frame do resultado do plano anterior", "Qué fotograma del resultado del plano anterior", "上一镜头结果中的哪一帧"],
  "the first shot uses only its references": ["o primeiro plano usa apenas suas referências", "el primer plano solo usa sus referencias", "第一个镜头仅使用自己的参考图"],
  "⚠ a camera cut starts this shot: history is skipped (it only carries a continuing take)": ["⚠ este plano começa após um corte de câmera: o histórico é ignorado (ele só funciona em uma tomada contínua)", "⚠ este plano empieza tras un corte de cámara: se omite el historial (solo sirve para una toma continua)", "⚠ 此镜头从镜头切换开始：跳过历史帧（历史帧只适用于连续长镜头）"],
  "the end of the PREVIOUS shot's result continues into this one (queue loop: shots run in order). Works with every mask mode, crop and the duet canvas.": ["o final do resultado do plano ANTERIOR continua neste plano (loop em fila: planos executados em ordem). Funciona com todos os modos de máscara, recorte e tela duet.", "el final del resultado del plano ANTERIOR continúa en este (bucle en cola: los planos se ejecutan en orden). Funciona con todos los modos de máscara, recorte y lienzo duet.", "上一镜头结果的末尾延续到此镜头（队列循环按顺序运行）。支持所有蒙版模式、裁剪和 duet 画布。"],
  "uses a frame of the PREVIOUS shot's result (queue loop, or auto loop with BFS Shot H3 Duet)": ["usa um frame do resultado do plano ANTERIOR (loop em fila ou loop automático com BFS Shot H3 Duet)", "usa un fotograma del resultado del plano ANTERIOR (bucle en cola o automático con BFS Shot H3 Duet)", "使用上一镜头结果中的一帧（队列循环，或搭配 BFS Shot H3 Duet 的自动循环）"],
  "the points, on each shot's frame at the same relative position (a subject that stays in place)": ["os pontos em cada plano, na mesma posição relativa do frame (para um sujeito que permanece no lugar)", "los puntos en cada plano, en la misma posición relativa del fotograma (para un sujeto que no se desplaza)", "在每个镜头相同相对位置的帧上放置这些点（适用于位置不变的主体）"],
  "the shot's mask video (it covers only its own shot: usually you want a different one per shot)": ["o vídeo de máscara do plano (cobre apenas este plano; normalmente cada plano precisa de um vídeo diferente)", "el vídeo de máscara del plano (solo cubre este plano; normalmente se usa uno distinto por plano)", "此镜头的蒙版视频（仅覆盖此镜头；通常每个镜头需要不同的视频）"],
  "the generation mode (Full frame / Frame + paste / Mask only / Crop / Crop + mask) and its mask strength (shots with a mask)": ["o modo de geração (Quadro inteiro / Quadro + colar / Somente máscara / Recorte / Recorte + máscara) e a força da máscara", "el modo de generación (Fotograma completo / Fotograma + pegar / Solo máscara / Recorte / Recorte + máscara) y la intensidad de máscara", "生成模式（整帧/整帧+贴回/仅蒙版/裁剪/裁剪+蒙版）及蒙版强度"],
  "the {target} description": ["a descrição de {target}", "la descripción de {target}", "{target} 的描述"],
  "continuity (not on the first shot)": ["continuidade (exceto no primeiro plano)", "continuidad (excepto en el primer plano)", "连续性（第一个镜头除外）"],
  "shots after this": ["planos depois deste", "planos posteriores a este", "此镜头之后的镜头"],
  "previous shot (←)": ["plano anterior (←)", "plano anterior (←)", "上一镜头（←）"],
  "📝 Pictures & prompt": ["📝 Imagens e prompt", "📝 Imágenes y prompt", "📝 图片和提示词"],
  "Pictures & prompt": ["Imagens e prompt", "Imágenes y prompt", "图片和提示词"],
  "every shot uses these unless it has its own": ["todos os planos usam estas, a menos que tenham as próprias", "todos los planos usan estas salvo que tengan las suyas", "所有镜头都使用这些，除非有自己的设置"],
  "faded = the global ones (Pictures & prompt tab)": ["esmaecidas = as globais (aba Imagens e prompt)", "atenuadas = las globales (pestaña Imágenes y prompt)", "半透明 = 全局图片（图片和提示词标签页）"],
  "Click + or drop images. Placeholders: ": ["Clique em + ou solte imagens. Marcadores: ", "Haz clic en + o suelta imágenes. Marcadores: ", "点击 + 或拖入图片。占位符："],
  " what changes (per shot) · ": [" o que muda (por plano) · ", " lo que cambia (por plano) · ", " 要改变的对象（逐镜头）· "],
  " picture descriptions · ": [" descrições das imagens · ", " descripciones de las imágenes · ", " 图片描述 · "],
  " the VLM's shot description · ": [" descrição do plano pelo VLM · ", " descripción del plano por el VLM · ", " VLM 的镜头描述 · "],
  " the setting picture. A connected prompt input overrides this prompt.": [" a imagem de cenário. Uma entrada prompt conectada substitui este prompt.", " la imagen de escenario. Una entrada prompt conectada sustituye este prompt.", " 场景图片。连接的 prompt 输入会覆盖此提示词。"],
  "What changes": ["O que muda", "Qué cambia", "要改变的内容"],
  "the selection, its picture and how it is generated": ["a seleção, a imagem dela e como é gerada", "la selección, su imagen y cómo se genera", "选区、对应图片以及生成方式"],
  "+ Add subject": ["+ Adicionar sujeito", "+ Añadir sujeto", "+ 添加对象"],
  "add pictures above to give each subject its own": ["adicione imagens acima para dar uma a cada sujeito", "añade imágenes arriba para dar una a cada sujeto", "在上方添加图片，为每个对象各分配一张"],
  "Colored Mark needs a selection: a text, points or a mask video.": ["A Marca colorida precisa de uma seleção: texto, pontos ou vídeo de máscara.", "La Marca de color necesita una selección: texto, puntos o vídeo de máscara.", "彩色标记需要选区：文本、标记点或蒙版视频。"],
  "what to select, in English: woman in pink top, red car, black dog… (Enter = preview)": ["o que selecionar, em inglês: woman in pink top, red car, black dog… (Enter = prévia)", "qué seleccionar, en inglés: woman in pink top, red car, black dog… (Enter = vista previa)", "要选择的对象（英文）：woman in pink top、red car、black dog…（回车 = 预览）"],
  "what to select, e.g. the man on the left": ["o que selecionar, ex.: the man on the left", "qué seleccionar, p. ej. the man on the left", "要选择的对象，例如 the man on the left"],
  "remove this subject": ["remover este sujeito", "quitar este sujeto", "移除此对象"],
  "or a mask video": ["ou um vídeo de máscara", "o un vídeo de máscara", "或蒙版视频"],
  "— none —": ["— nenhum —", "— ninguno —", "— 无 —"],
  "— none: the whole-video mask —": ["— nenhum: a máscara do vídeo inteiro —", "— ninguno: la máscara de todo el vídeo —", "— 无：整段视频蒙版 —"],
  "No selection of its own: this shot uses the whole-video mask (Mask settings, below).": ["Sem seleção própria: este plano usa a máscara do vídeo inteiro (Configurações de máscara, abaixo).", "Sin selección propia: este plano usa la máscara de todo el vídeo (Ajustes de máscara, abajo).", "没有单独选区：此镜头使用整段视频蒙版（见下方蒙版设置）。"],
  "Generate": ["Gerar", "Generar", "生成"],
  "The whole frame is regenerated.": ["O quadro inteiro é regenerado.", "Se regenera el fotograma entero.", "重新生成整个画面。"],
  "The whole frame is generated; BFS Shot Join pastes only the subject back onto the original (background exact).": ["O quadro inteiro é gerado; o BFS Shot Join cola só o sujeito sobre o original (fundo exato).", "Se genera el fotograma entero; BFS Shot Join pega solo el sujeto sobre el original (fondo exacto).", "生成整个画面；BFS Shot Join 只把对象贴回原视频（背景不变）。"],
  "Only the selection is regenerated; the rest of the frame stays exactly as it was.": ["Só a seleção é regenerada; o resto do quadro fica exatamente igual.", "Solo se regenera la selección; el resto del fotograma queda igual.", "只重新生成选区，画面其余部分保持不变。"],
  "A box around the selection is regenerated at full size (more detail for a small subject), then pasted back.": ["Uma caixa em volta da seleção é regenerada em tamanho cheio (mais detalhe para um sujeito pequeno) e colada de volta.", "Se regenera una caja alrededor de la selección a tamaño completo (más detalle para un sujeto pequeño) y se pega de vuelta.", "以完整尺寸重新生成选区周围的框（小对象细节更多），再贴回原处。"],
  "Inside the crop, only the selection is regenerated: the most detail, background kept.": ["Dentro do recorte, só a seleção é regenerada: o máximo de detalhe, com o fundo mantido.", "Dentro del recorte, solo se regenera la selección: el máximo detalle, con el fondo intacto.", "在裁剪框内只重新生成选区：细节最多，背景保留。"],
  "what is replaced, for {target} in the prompt (e.g. the young woman in a pink top)": ["o que é substituído, para {target} no prompt (ex.: the young woman in a pink top)", "lo que se sustituye, para {target} en el prompt (p. ej. the young woman in a pink top)", "被替换的对象，用于提示词中的 {target}（例如 the young woman in a pink top）"],
  "🧑 Describe": ["🧑 Descrever", "🧑 Describir", "🧑 描述"],
  "select: ": ["selecionar: ", "seleccionar: ", "选择："],
  "Use": ["Usar", "Usar", "使用"],
  "Mask settings (every shot)": ["Configurações de máscara (todos os planos)", "Ajustes de máscara (todos los planos)", "蒙版设置（所有镜头）"],
  "SAM 3, crop and paste-back": ["SAM 3, recorte e colagem", "SAM 3, recorte y pegado", "SAM 3、裁剪和贴回"],
  "grow the mask": ["aumenta a máscara", "amplía la máscara", "扩大蒙版"],
  "less flicker": ["menos cintilação", "menos parpadeo", "减少闪烁"],
  "matches of the text tracked": ["correspondências do texto rastreadas", "coincidencias del texto rastreadas", "跟踪的文本匹配数"],
  "crop modes": ["modos de recorte", "modos de recorte", "裁剪模式"],
  "Fill the selection text": ["Preencher o texto da seleção", "Rellenar el texto de selección", "填写选区文本"],
  "suggests each shot's selection and a description": ["sugere a seleção e uma descrição de cada plano", "sugiere la selección y una descripción de cada plano", "为每个镜头建议选区和描述"],
  "the text of {details} / {details@Picture n}": ["o texto de {details} / {details@Picture n}", "el texto de {details} / {details@Picture n}", "{details} / {details@Picture n} 的文本"],
  "Connect a Qwen3-VL to the planner's vlm input and run the workflow once: then these buttons work.": ["Conecte um Qwen3-VL à entrada vlm do planner e execute o workflow uma vez: depois estes botões funcionam.", "Conecta un Qwen3-VL a la entrada vlm del planner y ejecuta el workflow una vez: entonces estos botones funcionan.", "把 Qwen3-VL 连接到 planner 的 vlm 输入并运行一次工作流，之后这些按钮即可使用。"],
  "More: cut detector, frame grid, fps": ["Mais: detector de cortes, grade de frames, fps", "Más: detector de cortes, cuadrícula de fotogramas, fps", "更多：切点检测器、帧网格、fps"],
  "only the first N shots": ["só os N primeiros planos", "solo los primeros N planos", "仅前 N 个镜头"],
  "disabled": ["desativado", "desactivado", "已禁用"],
  "Choose a video, set the range and split it into shots.": ["Escolha um vídeo, defina o trecho e divida em planos.", "Elige un vídeo, define el rango y divídelo en planos.", "选择视频，设置范围并拆分为镜头。"],
  "Add the pictures and the prompt every shot uses. 🤖 Auto prompt can write the prompts for you.": ["Adicione as imagens e o prompt que todos os planos usam. O 🤖 Auto prompt pode escrever os prompts por você.", "Añade las imágenes y el prompt que usan todos los planos. 🤖 Auto prompt puede escribir los prompts por ti.", "添加所有镜头使用的图片和提示词。🤖 自动提示词可以替你写提示词。"],
  "Check each shot: its pictures, what changes (a text or points), and how it is generated.": ["Revise cada plano: as imagens, o que muda (texto ou pontos) e como é gerado.", "Revisa cada plano: sus imágenes, qué cambia (texto o puntos) y cómo se genera.", "检查每个镜头：图片、要改变的内容（文本或标记点）以及生成方式。"],
  "Colored Mark paints each selected subject with the color of the picture it becomes. Add one row per subject.": ["A Marca colorida pinta cada sujeito selecionado com a cor da imagem em que ele se transforma. Adicione uma linha por sujeito.", "La Marca de color pinta cada sujeto seleccionado con el color de la imagen en la que se convierte. Añade una fila por sujeto.", "彩色标记用目标图片的颜色涂抹每个选中的对象。每个对象添加一行。"],
  "Fix the warnings, then Queue. Stored shots can be redone one by one.": ["Corrija os avisos e coloque na fila. Planos salvos podem ser refeitos um a um.", "Corrige los avisos y pon en cola. Los planos guardados se pueden rehacer uno a uno.", "修正警告后加入队列。已保存的镜头可以逐个重做。"],
  "Frame + paste": ["Quadro + colagem", "Fotograma + pegado", "整帧 + 贴回"],
  "Mask only": ["Só a máscara", "Solo la máscara", "仅蒙版"],
  "Crop": ["Recorte", "Recorte", "裁剪"],
  "Crop + mask": ["Recorte + máscara", "Recorte + máscara", "裁剪 + 蒙版"],
  "runs": ["executa", "se ejecuta", "运行"],
  "translucent": ["translúcida", "translúcida", "半透明"],
  "hatching + contour": ["hachura + contorno", "rayado + contorno", "斜线 + 轮廓"],
  "normal map (needs normals)": ["normal map (precisa de normals)", "normal map (necesita normals)", "法线贴图（需要 normals）"],
  "Normal map: run MoGe on BFS Shot Unpack's frames (Render MoGe Geometry, normal_directx) and connect it to BFS Shot H3 Conditioning → normals. The preview shows a tint.": ["Normal map: rode o MoGe nos frames do BFS Shot Unpack (Render MoGe Geometry, normal_directx) e conecte em BFS Shot H3 Conditioning → normals. A prévia mostra só a cor.", "Normal map: ejecuta MoGe sobre los fotogramas de BFS Shot Unpack (Render MoGe Geometry, normal_directx) y conéctalo a BFS Shot H3 Conditioning → normals. La vista previa muestra solo el color.", "法线贴图：对 BFS Shot Unpack 的帧运行 MoGe（Render MoGe Geometry，normal_directx），并连接到 BFS Shot H3 Conditioning → normals。预览只显示颜色。"],
  "📍 MolmoPoint is connected: a text selection is pointed by MolmoPoint, then SAM 3 tracks it.": ["📍 MolmoPoint conectado: a seleção por texto é apontada pelo MolmoPoint e o SAM 3 rastreia a partir daí.", "📍 MolmoPoint conectado: la selección por texto la señala MolmoPoint y SAM 3 la sigue.", "📍 已连接 MolmoPoint：文本选区先由 MolmoPoint 定位，再由 SAM 3 跟踪。"],
};
export function t(value) {
  const s = String(value ?? ""), lang = uiLanguage.value;
  if (lang === "en") return s;
  const trimmed = s.trim(), entry = TEXT[s] || TEXT[trimmed];
  if (entry) return `${s.slice(0, s.indexOf(trimmed))}${entry[{ pt: 0, es: 1, zh: 2 }[lang]]}${s.slice(s.indexOf(trimmed) + trimmed.length)}`;
  let m;
  if ((m = s.match(/^Subject (\d+)$/))) return `${lang === "pt" ? "Sujeito" : lang === "es" ? "Sujeto" : "对象"} ${m[1]}`;
  if ((m = s.match(/^Mask strength (.+)$/))) return lang === "pt" ? `Força da máscara ${m[1]}` : lang === "es" ? `Fuerza de máscara ${m[1]}` : `蒙版强度 ${m[1]}`;
  if ((m = s.match(/^opacity (.+)$/))) return lang === "pt" ? `opacidade ${m[1]}` : lang === "es" ? `opacidad ${m[1]}` : `不透明度 ${m[1]}`;
  if ((m = s.match(/^Shot #(\d+)( · points)?$/))) return `${t("Shot")} #${m[1]}${m[2] ? ` · ${lang === "pt" ? "pontos" : lang === "es" ? "puntos" : "点"}` : ""}`;
  if ((m = s.match(/^(\d+) shot\(s\) will run$/))) return lang === "pt" ? `${m[1]} plano(s) serão executados` : lang === "es" ? `se ejecutarán ${m[1]} plano(s)` : `将运行 ${m[1]} 个镜头`;
  if ((m = s.match(/^(\d+) suggestions$/))) return lang === "pt" ? `${m[1]} sugestões` : lang === "es" ? `${m[1]} sugerencias` : `${m[1]} 条建议`;
  if ((m = s.match(/^(\d+) shot\(s\)$/))) return lang === "pt" ? `${m[1]} plano(s)` : lang === "es" ? `${m[1]} plano(s)` : `${m[1]} 个镜头`;
  if ((m = s.match(/^(\d+) warning\(s\) · (\d+) note\(s\)$/))) return lang === "pt" ? `${m[1]} aviso(s) · ${m[2]} observação(ões)` : lang === "es" ? `${m[1]} aviso(s) · ${m[2]} nota(s)` : `${m[1]} 条警告 · ${m[2]} 条备注`;
  if ((m = s.match(/^generate at (\d+)×(\d+)$/))) return lang === "pt" ? `gerar em ${m[1]}×${m[2]}` : lang === "es" ? `generar a ${m[1]}×${m[2]}` : `生成尺寸 ${m[1]}×${m[2]}`;
  if ((m = s.match(/^timeline (\S+) fps · (\d+) frames$/))) return lang === "pt" ? `linha do tempo ${m[1]} fps · ${m[2]} frames` : lang === "es" ? `línea de tiempo ${m[1]} fps · ${m[2]} fotogramas` : `时间轴 ${m[1]} fps · ${m[2]} 帧`;
  if ((m = s.match(/^from the start frame · (\d+) frames$/))) return lang === "pt" ? `a partir do frame inicial · ${m[1]} frames` : lang === "es" ? `desde el fotograma inicial · ${m[1]} fotogramas` : `从起始帧开始 · ${m[1]} 帧`;
  if ((m = s.match(/^🔊 audio · (.+) · (\d+) ch$/))) return lang === "pt" ? `🔊 áudio · ${m[1]} · ${m[2]} canais` : lang === "es" ? `🔊 audio · ${m[1]} · ${m[2]} canales` : `🔊 音频 · ${m[1]} · ${m[2]} 声道`;
  if ((m = s.match(/^🔇 (.+) → silent track$/))) return lang === "pt" ? `🔇 ${m[1]} → faixa silenciosa` : lang === "es" ? `🔇 ${m[1]} → pista silenciosa` : `🔇 ${m[1]} → 静音音轨`;
  if ((m = s.match(/^Sensitivity ([\d.]+)$/))) return lang === "pt" ? `Sensibilidade ${m[1]}` : lang === "es" ? `Sensibilidad ${m[1]}` : `灵敏度 ${m[1]}`;
  if ((m = s.match(/^= (\d+) frames \(([\d.]+) s\)$/))) return lang === "pt" ? `= ${m[1]} frames (${m[2]} s)` : lang === "es" ? `= ${m[1]} fotogramas (${m[2]} s)` : `= ${m[1]} 帧（${m[2]} 秒）`;
  if ((m = s.match(/^generate (\d+) frames$/))) return lang === "pt" ? `gerar ${m[1]} frames` : lang === "es" ? `generar ${m[1]} fotogramas` : `生成 ${m[1]} 帧`;
  if ((m = s.match(/^(\d+) too long \(> (\d+) f\)$/))) return lang === "pt" ? `${m[1]} acima do limite (> ${m[2]} frames)` : lang === "es" ? `${m[1]} demasiado largos (> ${m[2]} fotogramas)` : `${m[1]} 个镜头过长（> ${m[2]} 帧）`;
  if ((m = s.match(/^boundary @ (\d+)$/))) return lang === "pt" ? `divisão no frame ${m[1]}` : lang === "es" ? `límite en el fotograma ${m[1]}` : `分界位于第 ${m[1]} 帧`;
  if ((m = s.match(/^frame (\d+) · (.+)$/))) return lang === "pt" ? `frame ${m[1]} · ${m[2]}` : lang === "es" ? `fotograma ${m[1]} · ${m[2]}` : `第 ${m[1]} 帧 · ${m[2]}`;
  if ((m = s.match(/^(\d+)\/(\d+) shots · ([\d.]+)s$/))) return lang === "pt" ? `${m[1]}/${m[2]} planos · ${m[3]} s` : lang === "es" ? `${m[1]}/${m[2]} planos · ${m[3]} s` : `${m[1]}/${m[2]} 个镜头 · ${m[3]} 秒`;
  if ((m = s.match(/^(\d+) warning\(s\)$/))) return lang === "pt" ? `${m[1]} aviso(s)` : lang === "es" ? `${m[1]} aviso(s)` : `${m[1]} 条警告`;
  if ((m = s.match(/^(\d+) point\(s\) at frame (\d+)$/))) return lang === "pt" ? `${m[1]} ponto(s) no frame ${m[2]}` : lang === "es" ? `${m[1]} punto(s) en el fotograma ${m[2]}` : `第 ${m[2]} 帧有 ${m[1]} 个标记点`;
  if ((m = s.match(/^mask covers ([\d.]+)%(.+)$/))) return lang === "pt" ? `máscara cobre ${m[1]}%${m[2]}` : lang === "es" ? `la máscara cubre ${m[1]}%${m[2]}` : `蒙版覆盖 ${m[1]}%${m[2]}`;
  if ((m = s.match(/^last (\d+) frames \(([\d.]+) s\)$/))) return lang === "pt" ? `últimos ${m[1]} frames (${m[2]} s)` : lang === "es" ? `últimos ${m[1]} fotogramas (${m[2]} s)` : `最后 ${m[1]} 帧（${m[2]} 秒）`;
  if ((m = s.match(/^Play or scrub the video \(or click the timeline\), then ⇤ \/ ⇥\. (\d+) frames \(([\d.]+) s\) of (\d+): only this part is split, analysed, generated and joined\.$/)))
    return lang === "pt" ? `Reproduza ou avance o vídeo (ou clique na linha do tempo) e use ⇤ / ⇥. ${m[1]} frames (${m[2]} s) de ${m[3]}: somente este trecho será dividido, analisado, gerado e unido.` : lang === "es" ? `Reproduce o desplaza el vídeo (o haz clic en la línea de tiempo) y usa ⇤ / ⇥. ${m[1]} fotogramas (${m[2]} s) de ${m[3]}: solo se dividirá, analizará, generará y unirá este tramo.` : `播放或拖动视频（也可点击时间轴），然后按 ⇤ / ⇥。${m[1]} 帧（${m[2]} 秒），共 ${m[3]} 帧：仅拆分、分析、生成并合并此片段。`;
  return s;
}
function localizeProps(props) {
  if (!props || typeof props !== "object") return props;
  let changed = false; const out = { ...props };
  for (const k of ["title", "placeholder", "aria-label"]) if (typeof out[k] === "string") { const v = t(out[k]); changed ||= v !== out[k]; out[k] = v; }
  return changed ? out : props;
}
function localizeChildren(value) {
  if (typeof value === "string") return t(value);
  if (Array.isArray(value)) return value.map(localizeChildren);
  return value;
}
// All view modules use this h wrapper, so labels, placeholders, tooltips and section titles stay in sync.
export function h(type, props, children) {
  const hasChildrenArg = arguments.length >= 3;
  if (!hasChildrenArg && props?.__v_isVNode) return vueH(type, null, props);
  const p = localizeProps(props);
  if (hasChildrenArg) return vueH(type, p, localizeChildren(children));
  if (typeof props === "string" || Array.isArray(props)) return vueH(type, localizeChildren(props));
  return vueH(type, p);
}
// Use for literal labels that must keep their native spelling (for example language names).
export const rawH = (...args) => vueH(...args);

export const GRIDS = { "H3 (17n+5)": [17, 5], "LTX / Wan (8n+1)": [8, 1], "Wan (4n+1)": [4, 1], "any": [1, 0] };
export const DEFAULTS = {
  video: "", fps: 24, grid: "H3 (17n+5)", mode: "shots", max_s: 4.5, min_s: 1.0, sensitivity: 0.5,
  max_parts: 0, max_total_s: 0, start_s: 0, end_s: 0, start_f: null, end_f: null, track: null, bounds: [], segs: [], global_refs: [], global_ref: "", global_ref2: "", global_prompt: "", global_color_guide_ref1: "#00ff00", global_color_guide_ref2: "#9b30ff", mask_video: "",
  megapixels: 0.5, multiple: 32, generation_size_mode: "megapixels", custom_width: 1024, custom_height: 1024,
  second_pass: { enabled: false, mode: "scale", scale: 2, width: 1280, height: 704, megapixels: 1, align: 32 },
  detector: "adaptive", run: "recursive", between_shots: "keep", shot_cache: "output", shot_storage: "lossless", unload_before_join: true, auto_continue: true,
  skip_fill: "original", audio_mode: "auto", ref_details: {}, picture_details: {}, auto_prompt: {}, auto_prompts: {}, mask_cfg: {}, vlm_cfg: {},
};

// global mask settings (same defaults as the server's DEFAULT_MASK)
export const MASK_DEFAULTS = { threshold: 0.5, max_objects: 4, invert: false, fill_holes: true, temporal_expand: 2, blockify: 0,
                               padding: 0.15, expand: 16, feather: 12, paste: "box" };

export const VLM_DEFAULTS = { enabled: false, frames: 3, max_tokens: 1024, auto_segment: true, auto_shot: true, instruction: "",
                              describe_preset: "short", describe_custom: "", fit_prompt: true, temperature: 0, seed: 0,
                              image_max_side: 1024 };

// hover help for every setting (the 📖 Guide button opens the full documentation)
export const TIPS = {
  "Mode": "Camera cuts: shots start at the detected cuts, long ones are split evenly. Fixed length: equal parts no longer than the maximum.",
  "Detector": "Cut detector. PySceneDetect adaptive works on most footage (fast camera moves don't trigger it); content is stricter on hard cuts; built-in needs no package.",
  "Sensitivity": "How easily a change counts as a cut. Higher finds more cuts (and more false ones).",
  "Frame grid": "Frame counts the model accepts. H3: 17n+5, LTX/Wan: 8n+1, Wan: 4n+1. Each shot is generated at the next valid length and trimmed back.",
  "Max seconds / shot": "Longest shot sent to the model. Keep it within what the model / LoRA was trained on (H3 body swap: about 4.5 s).",
  "Min seconds / shot": "Shorter shots merge into a neighbour when the merge still fits the maximum.",
  "Timeline fps": "The planner works on this frame rate (the source is resampled). Generated video and audio use it too.",
  "Max shots": "Only the first N shots (0 = all). Handy for quick tests.",
  "Megapixels": "Generation size: the source aspect ratio at this pixel area (0.5 ≈ 960×544 for 16:9). Cropped shots use it for the crop.",
  "Size multiple": "Width and height snap to this multiple (32 for H3 and most video models).",
  "Shot cache": "Where the generated shots are stored. Keep: ComfyUI's output folder (output/bfs_shotloop): they survive a restart, so a stopped run resumes and a single shot can be redone. Temp: ComfyUI empties it at every start.",
  "Shot storage": "How each stored shot is saved. Lossless (FFV1): exact, ~12x smaller than raw frames. Smaller / Smallest (H.264 RGB): about 20x / 40x smaller with a slight loss, which reaches the final video and the continuity head. Raw: exact, biggest, needs no ffmpeg.",
  "Between shots": "Keep models loaded: fastest. Free memory: after each shot, unload every model and empty the GPU, so the next shot starts clean (for low VRAM or when a later shot runs out of memory); models load again for every shot, so it is slower.",
  "Run": "One run, shot by shot: one Queue renders every shot in order, storing each on disk (continuity works, resumes after a failure). Queue loop: one shot per run, the panel re-queues (keep the tab open). Auto loop: every shot at once in memory (no continuity).",
  "Skipped shots in the output": "Shots that do not run: keep the original video there, or remove them (with their audio).",
  "Crop padding": "Context kept around the mask's box, as a % of the box. More context blends better; less gives the model more pixels on the subject.",
  "Expand": "Grow the mask by this many pixels: before pasting a cropped result back, and for the generation mask (inpaint 'only the mask'). Raise it when the new person is bigger than the old one (longer hair, wider body).",
  "Feather": "Soft edge of the paste-back, in pixels.",
  "Temporal expand": "Hold the mask this many frames before and after: removes flicker where SAM 3 drops a frame.",
  "Blockify": "Square blocks of this size (0 = off). 16 matches H3's latent grid, useful as an inpainting mask.",
  "Detection threshold": "SAM 3 score needed to keep a text match. Lower finds more (and more wrong) objects.",
  "Max objects": "How many matches of the text prompt are tracked together.",
  "Paste back": "Only the mask: the result replaces the masked pixels. The whole box: the full crop is pasted with feathered borders.",
  "Frames per shot": "How many frames of each shot the VLM looks at.",
  "Max tokens": "Longest VLM answer. Raise it for long descriptions.",
};

export const snapUp = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.ceil((n - o) / s);
};
export const snapDown = (n, grid) => {
  const [s, o] = GRIDS[grid] || GRIDS["H3 (17n+5)"]; n = Math.max(1, n | 0);
  if (s === 1) return n; if (n <= o) return o; return o + s * Math.floor((n - o) / s);
};
export const hue = i => `hsl(${(i * 47 + 200) % 360} 55% 46%)`;
export const fmtT = (f, fps) => { const s = f / fps; return `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, "0")}`; };
export const segKey = s => `${s.start}-${s.end}`;
// a shot has a mask: its own mask video, SAM 3 text or points, or the plan's mask video for the whole video
// (the planner's `mask` input is only known when the workflow runs)
export const hasMask = s => !!(s.mask?.video || s.mask?.text || (s.mask?.points || []).length || s.extMask);
export const maskSource = s => s.mask?.video ? "video" : (s.mask?.text || (s.mask?.points || []).length) ? "sam"
  : s.trackMask ? "track" : s.extMask ? "global" : "";

// ---- small view helpers
export const pill = (text, kind = "", title = "") => h("span", { class: ["pill", kind], title }, text);
export const hint = (text, style = "") => h("span", { class: "hint", style }, text);
export const fld = (label, input, help) => {
  const tip = Object.entries(TIPS).find(([k]) => String(label).startsWith(k))?.[1] || "";
  return h("div", { class: "fld", title: tip }, [h("label", tip ? [label, h("span", { class: "qm" }, " ⓘ")] : label), input,
    help ? h("div", { class: "hint" }, help) : null]);
};
export const check = (checked, onChange, label, title = "") =>
  h("label", { class: "chk", title }, [h("input", { type: "checkbox", checked: !!checked, onChange: e => onChange(e.target.checked) }), label]);
export const select = (value, options, onChange, attrs = {}) => h("select", { value, onChange: e => onChange(e.target.value), ...attrs },
  options.map(o => h("option", { value: Array.isArray(o) ? o[0] : o }, Array.isArray(o) ? o[1] : o)));
// a titled section inside a card; `open` makes it a collapsible <details>
// what the user opened or closed wins over a section's default: re-renders (a timer, a status line) no longer snap a
// section back shut right after it was clicked open
const OPEN = reactive({});
export const section = (title, body, { sub = "", right = null, open = null, cls = "" } = {}) => {
  const head = [h("span", { class: "sect" }, title), sub ? hint(sub) : null, h("span", { class: "grow" }), right];
  if (open === null) return h("div", { class: ["sec", cls] }, [h("div", { class: "sech" }, head), ...[body].flat()]);
  const key = `${cls}|${typeof title === "string" ? title : ""}`;
  const isOpen = key in OPEN ? OPEN[key] : !!open;
  return h("details", { class: ["sec", cls], open: isOpen, onToggle: e => { OPEN[key] = e.target.open; } },
    [h("summary", { class: "sech" }, [h("span", { class: "caret" }, "▸"), ...head]), ...[body].flat()]);
};

// second pass size, the same arithmetic as bfs_shot_loop.second_pass_size (and Minimax H3 Latent Upscaler (3D)):
// target pixels, rounded to `align`, then to whole 16 px latent cells
export const secondPassSize = (w, h, sp) => {
  if (!w || !h) return { w: 0, h: 0 };
  let tw, th;
  if (sp.mode === "dims") { tw = +sp.width; th = +sp.height; }
  else if (sp.mode === "megapixels") { const px = +sp.megapixels * 1024 * 1024; th = Math.sqrt(px / (w / h)); tw = th * (w / h); }
  else { tw = w * +sp.scale; th = h * +sp.scale; }
  const al = Math.max(1, Math.round(+sp.align || 1)), r = x => Math.round(x / al) * al, c16 = x => Math.max(16, Math.round(x / 16) * 16);
  return { w: c16(r(tw)), h: c16(r(th)) };
};

// ---- reference pictures (<Picture 1..12>)
export const MAX_REFS = 100;
// a reference list of the plan: `refs` (plans from 1.113 on), else the old ref / ref2 pair; empty slots at the end dropped
export const refList = (refs, a = "", b = "") => {
  const out = (Array.isArray(refs) ? refs : [a, b]).map(x => (typeof x === "string" ? x : "")).slice(0, MAX_REFS);
  while (out.length && !out[out.length - 1]) out.pop();
  return out;
};
// the pictures a shot uses, in <Picture n> order: its own slot, else the global one at the same position; a slot with
// neither is left out (same rule as bfs_shot_loop.shot_ref_names)
export const shotRefNames = (own, glob) => {
  const out = [];
  for (let k = 0; k < Math.max(own.length, glob.length); k++) { const x = own[k] || glob[k]; if (x) out.push(x); }
  return out.slice(0, MAX_REFS);
};
// one slot per picture, then "+" (up to 12). Click: upload (the "+" takes several files at once); drop image files or a
// recent thumbnail on a slot to replace it, anywhere else in the box to add them. `fallback`: the global pictures a
// shot's empty slots use (shown faded). `extra(k, name)`: more controls under slot k.
export function refBox(c, target, own, fallback = [], extra = () => null) {
  const n = Math.max(own.length, fallback.length);
  const drop = k => ({ onDragover: e => e.preventDefault(),
                       onDrop: e => { e.preventDefault(); e.stopPropagation(); c.dropRefs(target, k, e); } });
  const slots = [];
  for (let k = 0; k < n; k++) {
    const name = own[k] || "", via = name ? "" : fallback[k] || "";
    slots.push(h("div", { class: "rslot" }, [
      h("div", { class: "refslot", ...drop(k),
        title: name ? `<Picture ${k + 1}>: ${name} (click to replace)` : via ? `<Picture ${k + 1}>: ${via} (global; click to use another picture in this shot)` : "click to upload",
        onClick: () => c.uploadRef(target, k) },
        name || via ? [h("img", { src: c.viewUrl(name || via), style: via ? "opacity:.45" : "" }), h("span", { class: "rnum" }, String(k + 1))]
          : [`⬆ ${k + 1}`]),
      h("div", { class: "rbtns" }, [name ? h("button", { title: target === "global" || k >= fallback.length ? "remove this picture"
        : "remove (use the global picture)", onClick: () => c.setRef(target, k, "") }, "✕") : null]),
      extra(k, name || via),
    ]));
  }
  if (n < MAX_REFS) slots.push(h("div", { class: "rslot" }, [
    h("div", { class: "refslot add", ...drop(n), title: `add pictures (up to ${MAX_REFS}): click to upload one or more, or drop image files here`,
      onClick: () => c.uploadRef(target, n, true) }, "+"),
    h("div", { class: "rbtns" }, []),
  ]));
  return h("div", { class: "refbox", ...drop(null) }, slots);
}
// chips under a prompt, one per picture: PictureN inserts <Picture N> at the cursor, 📝 inserts {details@Picture N}
// (the description of that picture alone). The prompt box and its chips sit in one .pbox.
export function promptChips(c, names, onSet) {
  if (!names.length) return null;
  const insert = (e, tag) => {
    e.preventDefault();   // mousedown: the prompt keeps its focus and cursor
    const ta = e.currentTarget.closest(".pbox")?.querySelector("textarea"); if (!ta) return;
    const v = ta.value, a = ta.selectionStart ?? v.length, b = ta.selectionEnd ?? v.length;
    const before = v.slice(0, a) + (a && !/\s$/.test(v.slice(0, a)) ? " " : "") + tag;
    onSet(before + v.slice(b));
    requestAnimationFrame(() => { ta.focus(); ta.setSelectionRange(before.length, before.length); });
  };
  return h("div", { class: "pchips" }, names.map((name, k) => h("span", { class: "pchip" }, [
    h("button", { class: "sm", title: `insert <Picture ${k + 1}> (${name})`, onMousedown: e => insert(e, `<Picture ${k + 1}>`) },
      [h("img", { src: c.viewUrl(name) }), `Picture${k + 1}`]),
    h("button", { class: "sm", title: `insert {details@Picture ${k + 1}}: the description of this picture (📝 Describe refs, or the VLM at run time)`,
      onMousedown: e => insert(e, `{details@Picture ${k + 1}}`) }, "📝"),
  ])));
}

// ---- Colored Mark: one color per picture (<Picture 1> first; the first seven are the colors of the first mark sets)
export const MARK_PALETTE = [["#00ff00", "green"], ["#9b30ff", "purple"], ["#ff2020", "red"], ["#168cff", "blue"],
  ["#ffe600", "yellow"], ["#ff00d4", "magenta"], ["#00e5c0", "cyan"], ["#ff8000", "orange"], ["#ffffff", "white"],
  ["#8b4513", "brown"], ["#ff9ec4", "pink"], ["#808000", "olive"]];
export const markName = color => (MARK_PALETTE.find(([v]) => v === String(color).toLowerCase()) || [0, color])[1];
// the mark color of <Picture k> (same rule as bfs_shot_loop.picture_color)
export const pictureColor = (plan, k) => {
  const c = Array.isArray(plan.picture_colors) ? plan.picture_colors[k - 1] : "";
  return /^#[0-9a-f]{6}$/i.test(c || "") ? c.toLowerCase() : MARK_PALETTE[(k - 1) % MARK_PALETTE.length][0];
};
// a shot's extra Colored Mark regions (the second person on): plans before 1.113 had one (text2 / points2 = picture 2)
export const markRegions = m => Array.isArray(m.color_guide_regions) ? m.color_guide_regions
  : (m.color_guide_text2 || (m.color_guide_points2 || []).length)
    ? [{ picture: 2, text: m.color_guide_text2 || "", points: m.color_guide_points2 || [], key: m.color_guide_key2 || 0 }] : [];

// ---- 🤖 Auto prompt: the VLM writes each shot's H3 prompt from the request in the prompt box
export const AUTO_DEFAULTS = { enabled: false, frames: 0, source: "guide", style: "h3" };
