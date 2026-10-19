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
  "🎞 Video": ["🎞 Vídeo", "🎞 Vídeo", "🎞 视频"], "🎬 Shots": ["🎬 Shots", "🎬 Planos", "🎬 镜头"],
  "👥 People & masks": ["👥 Pessoas e máscaras", "👥 Personas y máscaras", "👥 人物和蒙版"],
  "📝 Prompts & refs": ["📝 Prompts e referências", "📝 Prompts y referencias", "📝 提示词和参考图"],
  "▶ Run": ["▶ Executar", "▶ Ejecutar", "▶ 运行"], "📖 Guide": ["📖 Guia", "📖 Guía", "📖 指南"],
  "🧭 Tour": ["🧭 Tour", "🧭 Recorrido", "🧭 引导"], "Language": ["Idioma", "Idioma", "语言"],
  "Source video": ["Vídeo de origem", "Vídeo de origen", "源视频"], "Range": ["Trecho", "Rango", "范围"],
  "Timeline": ["Linha do tempo", "Línea de tiempo", "时间轴"], "zoom": ["zoom", "zoom", "缩放"],
  "Click a shot to select it (← → move between shots) · drag the white handles to move a boundary · double-click the shot bar to split · Delete removes the selected shot's cut": [
    "Clique em um plano para selecioná-lo (← → navega entre planos) · arraste as alças brancas para mover uma divisão · dê dois cliques na barra para dividir · Delete remove o corte do plano selecionado",
    "Haz clic en un plano para seleccionarlo (← → cambia de plano) · arrastra los controles blancos para mover un límite · haz doble clic en la barra para dividir · Delete elimina el corte del plano seleccionado",
    "点击镜头以选中（← → 切换镜头）· 拖动白色控点移动分界 · 双击镜头条进行拆分 · 按 Delete 删除所选镜头的切点",
  ],
  "camera cuts or fixed length, on the model's frame grid": ["cortes de câmera ou duração fixa, na grade de frames do modelo", "cortes de cámara o duración fija, según la cuadrícula de fotogramas del modelo", "按镜头切换或固定时长，并符合模型帧步长"],
  "Range length, seconds (0 = to the end)": ["Duração do trecho, em segundos (0 = até o fim)", "Duración del tramo, en segundos (0 = hasta el final)", "片段时长（秒，0 = 到结尾）"],
  "refresh the input folder": ["atualizar a pasta de entrada", "actualizar la carpeta de entrada", "刷新输入文件夹"],
  "Pick or upload a video to start": ["Escolha ou envie um vídeo para começar", "Elige o sube un vídeo para empezar", "选择或上传视频以开始"],
  "use the video’s audio": ["usar o áudio do vídeo", "usar el audio del vídeo", "使用视频音频"],
  "The outputs carry a silent track of the right length, so Create Video works": ["As saídas incluem uma faixa silenciosa com a duração correta para funcionar em Create Video", "Las salidas incluyen una pista silenciosa con la duración correcta para que funcione Create Video", "输出会带有时长正确的静音音轨，以便 Create Video 正常工作"],
  "Audio of the planner's and the join's outputs": ["Áudio das saídas do Planner e do Join", "Audio de las salidas del Planner y del Join", "Planner 和 Join 输出的音频"],
  "the range ends after the player's frame": ["o trecho termina depois do frame atual do player", "el tramo termina después del fotograma actual del reproductor", "片段在播放器当前帧之后结束"],
  "Play or scrub the video (or click the timeline), then ⇤ / ⇥. {…} frames ({…} s) of {…}: only this part is split, analysed, generated and joined.": ["Reproduza ou avance o vídeo (ou clique na linha do tempo) e use ⇤ / ⇥. {…} frames ({…} s) de {…}: somente este trecho será dividido, analisado, gerado e unido.", "Reproduce o desplaza el vídeo (o haz clic en la línea de tiempo) y usa ⇤ / ⇥. {…} fotogramas ({…} s) de {…}: solo se dividirá, analizará, generará y unirá este tramo.", "播放或拖动视频（也可点击时间轴），然后按 ⇤ / ⇥。共 {…}/{…} 帧（{…} 秒）：仅拆分、分析、生成并合并此片段。"],
  "15 s in one shot": ["15 s em um único plano", "15 s en un solo plano", "单镜头 15 秒"],
  "from the start frame · {…} frames": ["a partir do frame inicial · {…} frames", "desde el fotograma inicial · {…} fotogramas", "从起始帧开始 · {…} 帧"],
  "limit how many detected shots are used": ["limita quantos planos detectados serão usados", "limita cuántos planos detectados se usan", "限制使用的检测镜头数量"],
  "Auto split replaces the boundaries; per-shot settings stay with the shot at the same position.": ["A divisão automática substitui os limites; as configurações por plano permanecem na mesma posição.", "La división automática reemplaza los límites; los ajustes por plano se mantienen en la misma posición.", "自动拆分会替换分界；逐镜头设置保留在相同位置。"],
  "before the start: not planned": ["antes do início: fora do plano", "antes del inicio: no incluido", "起点之前：不处理"],
  "after the end: not planned": ["depois do fim: fora do plano", "después del final: no incluido", "终点之后：不处理"],
  "Split into shots": ["Dividir em planos", "Dividir en planos", "拆分镜头"], "Generation size": ["Tamanho da geração", "Tamaño de generación", "生成尺寸"],
  "Camera cuts": ["Cortes de câmera", "Cortes de cámara", "镜头切换"], "Fixed length": ["Duração fixa", "Duración fija", "固定时长"],
  "One shot (whole range)": ["Um plano (trecho inteiro)", "Un plano (rango completo)", "单个镜头（整个范围）"],
  "Mode": ["Modo", "Modo", "模式"], "Detector": ["Detector", "Detector", "检测器"], "Sensitivity": ["Sensibilidade", "Sensibilidad", "灵敏度"],
  "Frame grid": ["Grade de frames", "Múltiplo de fotogramas", "帧步长"], "Max seconds / shot": ["Máx. segundos por plano", "Máx. segundos por plano", "每个镜头最长秒数"],
  "Min seconds / shot": ["Mín. segundos por plano", "Mín. segundos por plano", "每个镜头最短秒数"], "Timeline fps": ["FPS da linha do tempo", "FPS de la línea de tiempo", "时间轴帧率"],
  "Megapixels": ["Megapixels", "Megapíxeles", "百万像素"], "Size multiple": ["Múltiplo do tamanho", "Múltiplo del tamaño", "尺寸倍数"],
  "Size mode": ["Modo de tamanho", "Modo de tamaño", "尺寸模式"], "Megapixel target": ["Alvo em megapixels", "Objetivo en megapíxeles", "百万像素目标"],
  "Original video": ["Vídeo original", "Vídeo original", "原视频"], "Custom dimensions": ["Dimensões personalizadas", "Dimensiones personalizadas", "自定义尺寸"],
  "Output size": ["Tamanho de saída", "Tamaño de salida", "输出尺寸"],
  "Custom width": ["Largura personalizada", "Ancho personalizado", "自定义宽度"], "Custom height": ["Altura personalizada", "Alto personalizado", "自定义高度"],
  "Image max side (VLM)": ["Lado máximo da imagem (VLM)", "Lado máximo de imagen (VLM)", "VLM 图像最长边"],
  "VLM image sizing": ["Redimensiona as imagens enviadas ao VLM, mantendo a proporção. O processador do modelo ainda pode redimensioná-las.", "Ajusta las imágenes enviadas al VLM manteniendo la proporción. El procesador del modelo puede volver a ajustarlas.", "调整发送给 VLM 的图像并保持比例。模型处理器仍可能再次缩放。"],
  "Original resolution is rounded to the size multiple for model compatibility.": ["A resolução original é arredondada ao múltiplo definido para compatibilidade com o modelo.", "La resolución original se redondea al múltiplo indicado para compatibilidad con el modelo.", "原始分辨率会取整到尺寸倍数，以兼容模型。"],
  "✂ Auto split": ["✂ Dividir automaticamente", "✂ División automática", "✂ 自动拆分"], "↻ Re-analyse": ["↻ Analisar novamente", "↻ Volver a analizar", "↻ 重新分析"],
  "Reference & prompt": ["Referência e prompt", "Referencia y prompt", "参考图和提示词"], "Who is replaced": ["Quem será substituído", "Quién se reemplaza", "替换对象"],
  "Continuity": ["Continuidade", "Continuidad", "连续性"], "Copy to other shots": ["Copiar para outros planos", "Copiar a otros planos", "复制到其他镜头"],
  "Global reference & prompt": ["Referência e prompt globais", "Referencia y prompt globales", "全局参考图和提示词"],
  "Describe references ({details})": ["Descrever referências ({details})", "Describir referencias ({details})", "描述参考图（{details}）"],
  "Ref 1 description ({details})": ["Descrição da ref. 1 ({details})", "Descripción de ref. 1 ({details})", "参考图 1 描述（{details}）"],
  "Ref 2 description ({details2})": ["Descrição da ref. 2 ({details2})", "Descripción de ref. 2 ({details2})", "参考图 2 描述（{details2}）"],
  "no description yet: Describe refs, write your own, or leave it to the VLM at run time": ["sem descrição: use Descrever referências, escreva uma descrição ou deixe o VLM gerar durante a execução", "sin descripción: pulsa Describir referencias, escribe una o deja que el VLM la genere al ejecutar", "暂无描述：点击“描述参考图”、自行填写，或在运行时由 VLM 生成"],
  "VLM": ["VLM", "VLM", "视觉语言模型"], "Use the VLM when the workflow runs": ["Usar o VLM ao executar o workflow", "Usar el VLM al ejecutar el flujo", "运行工作流时使用 VLM"],
  "Fill the mask text": ["Preencher o texto da máscara", "Completar el texto de la máscara", "填写蒙版描述"],
  "Fill {shot} in the prompt": ["Preencher {shot} no prompt", "Completar {shot} en el prompt", "在提示词中填入 {shot}"],
  "Frames per shot": ["Frames por plano", "Fotogramas por plano", "每个镜头的帧数"], "Max tokens": ["Máx. tokens", "Máx. tokens", "最大 token 数"],
  "🤖 Analyse shots": ["🤖 Analisar planos", "🤖 Analizar planos", "🤖 分析镜头"], "📝 Describe refs": ["📝 Descrever referências", "📝 Describir referencias", "📝 描述参考图"],
  "🎨 Colored Mark": ["🎨 Marcação colorida", "🎨 Marca de color", "🎨 彩色标记"], "Colored Mark": ["Marcação colorida", "Marca de color", "彩色标记"], "Ref 1 color": ["Cor da ref. 1", "Color de ref. 1", "参考图 1 的颜色"],
  "Ref 2": ["Ref. 2", "Ref. 2", "参考图 2"], "+ Add a second person (optional)": ["+ Adicionar segunda pessoa (opcional)", "+ Añadir segunda persona (opcional)", "+ 添加第二个人（可选）"],
  "Second person · Ref 2": ["Segunda pessoa · Ref. 2", "Segunda persona · Ref. 2", "第二个人 · 参考图 2"],
  "Describe the second person for SAM 3": ["Descreva a segunda pessoa para o SAM 3", "Describe la segunda persona para SAM 3", "为 SAM 3 描述第二个人"],
  "👁 Preview": ["👁 Prévia", "👁 Vista previa", "👁 预览"], "More options": ["Mais opções", "Más opciones", "更多选项"],
  "Keep ref 2 color consistent across shots (optional)": ["Manter a cor da ref. 2 nos outros planos (opcional)", "Mantener el color de ref. 2 en otros planos (opcional)", "在其他镜头中保持参考图 2 的颜色（可选）"],
  "Ref 1 identity": ["Identidade da ref. 1", "Identidad de ref. 1", "参考图 1 的身份"], "Ref 2 identity": ["Identidade da ref. 2", "Identidad de ref. 2", "参考图 2 的身份"],
  "Choose manually": ["Escolher manualmente", "Elegir manualmente", "手动选择"], "↩ cast color": ["↩ cor do elenco", "↩ color del reparto", "↩ 使用角色颜色"],
  "Opacity": ["Opacidade", "Opacidad", "不透明度"], "The default keeps movement and expression visible.": ["O padrão mantém o movimento e a expressão visíveis.", "El valor predeterminado mantiene visibles el movimiento y la expresión.", "默认值会保留动作和表情。"],
  "Full frame": ["Quadro inteiro", "Fotograma completo", "整帧"], "🧩 Frame + paste": ["🧩 Quadro + colar", "🧩 Fotograma + pegar", "🧩 整帧生成并贴回"],
  "🎭 Mask only": ["🎭 Somente máscara", "🎭 Solo máscara", "🎭 仅蒙版"], "✂ Crop": ["✂ Recorte", "✂ Recorte", "✂ 裁剪"], "✂🎭 Crop + mask": ["✂🎭 Recorte + máscara", "✂🎭 Recorte + máscara", "✂🎭 裁剪并蒙版"],
  "People & masks": ["Pessoas e máscaras", "Personas y máscaras", "人物和蒙版"], "Cast": ["Elenco", "Reparto", "角色"],
  "Mask & crop settings": ["Configurações de máscara e recorte", "Ajustes de máscara y recorte", "蒙版和裁剪设置"],
    "Green": ["Verde", "Verde", "绿色"], "Purple": ["Roxo", "Morado", "紫色"],
  "Red": ["Vermelho", "Rojo", "红色"], "Blue": ["Azul", "Azul", "蓝色"], "Yellow": ["Amarelo", "Amarillo", "黄色"], "Magenta": ["Magenta", "Magenta", "品红"], "Cyan": ["Ciano", "Cian", "青色"],
  "Checks": ["Verificações", "Comprobaciones", "检查"], "Run": ["Executar", "Ejecutar", "运行"], "Filters": ["Filtros", "Filtros", "筛选"],
  "Auto loop (one run)": ["Loop automático (uma execução)", "Bucle automático (una ejecución)", "自动循环（一次运行）"],
  "Queue loop (one shot per run)": ["Fila (um plano por execução)", "Cola (un plano por ejecución)", "队列循环（每次运行一个镜头）"],
  "Skipped shots in the output": ["Planos ignorados na saída", "Planos omitidos en la salida", "输出中跳过的镜头"], "Keep original video": ["Manter vídeo original", "Mantener vídeo original", "保留原视频"],
  "Remove them": ["Removê-los", "Quitarlos", "移除"], "Max shots (0 = all)": ["Máx. planos (0 = todos)", "Máx. planos (0 = todos)", "镜头上限（0 = 全部）"],
  "Max total seconds (0 = all)": ["Máx. duração total (0 = tudo)", "Duración total máx. (0 = todo)", "总时长上限（0 = 全部）"],
  "Auto-queue the next shot": ["Enfileirar próximo plano automaticamente", "Añadir automáticamente el siguiente plano a la cola", "自动将下一个镜头加入队列"],
  "Needs a person": ["Precisa ter uma pessoa", "Debe haber una persona", "需要有人物"], "Needs a face": ["Precisa ter um rosto", "Debe haber un rostro", "需要有人脸"],
  "Skip dark / fades": ["Ignorar cenas escuras / fades", "Omitir escenas oscuras / fundidos", "跳过暗场/淡入淡出"], "Skip static shots": ["Ignorar planos estáticos", "Omitir planos estáticos", "跳过静止镜头"],
  "👤 Analyse people & faces": ["👤 Analisar pessoas e rostos", "👤 Analizar personas y rostros", "👤 分析人物和人脸"],
  "Video": ["Vídeo", "Vídeo", "视频"], "Shots": ["Planos", "Planos", "镜头"], "People": ["Pessoas", "Personas", "人物"], "Prompts": ["Prompts", "Prompts", "提示词"],
  "Back": ["Voltar", "Atrás", "上一步"], "Next": ["Avançar", "Siguiente", "下一步"], "Close tour": ["Fechar tour", "Cerrar recorrido", "关闭引导"],
  "Shot Planner tour": ["Tour do Shot Planner", "Recorrido de Shot Planner", "Shot Planner 引导"], "Shot": ["Plano", "Plano", "镜头"],
  "This tour is optional. Use Next to visit each main tab; close it whenever you like.": ["Este tour é opcional. Use Avançar para conhecer cada aba; feche quando quiser.", "Este recorrido es opcional. Pulsa Siguiente para visitar cada pestaña; puedes cerrarlo cuando quieras.", "此引导为可选。点击下一步浏览各个主选项卡，也可随时关闭。"],
  "Choose a video, set the frame range, split at camera cuts, then set the generation size.": ["Escolha um vídeo e o trecho, divida nos cortes de câmera e defina o tamanho da geração.", "Elige un vídeo y un rango, divídelo en cortes de cámara y define el tamaño de generación.", "选择视频和范围，按镜头切换拆分，然后设置生成尺寸。"],
  "Review each shot, assign its references and prompt, and choose how the result is applied.": ["Revise cada plano, associe referências e prompt e escolha como aplicar o resultado.", "Revisa cada plano, asigna referencias y prompt, y elige cómo aplicar el resultado.", "检查每个镜头，设置参考图和提示词，并选择结果的应用方式。"],
  "Track detected face identities to filter or split shots. Set masks here or per shot.": ["Rastreie identidades faciais para filtrar ou dividir planos. Configure as máscaras aqui ou por plano.", "Sigue identidades faciales para filtrar o dividir planos. Configura máscaras aquí o por plano.", "跟踪人脸身份以筛选或拆分镜头。可在此处或逐镜头设置蒙版。"],
  "Set shared references, prompts and optional VLM assistance. Written prompts are kept exactly as authored.": ["Defina referências, prompts e assistência opcional do VLM. Os prompts escritos são preservados como estão.", "Configura referencias, prompts y asistencia opcional del VLM. Los prompts escritos se conservan tal cual.", "设置共享参考图、提示词和可选的 VLM 辅助。提示词会保持原样。"],
  "Check warnings, choose auto or queue loop, and set filters or quick-run limits.": ["Confira avisos, escolha loop automático ou fila e configure filtros e limites de teste.", "Revisa avisos, elige bucle automático o cola y configura filtros y límites de prueba.", "查看警告，选择自动循环或队列，并设置筛选条件和快速运行限制。"],
  "Color the selected region in the source video as a translucent visual marker. It does not create or enable an inpainting mask.": ["Colorir a região selecionada no vídeo de origem como uma marca translúcida. Isso não cria nem ativa uma máscara de inpainting.", "Colorea la región seleccionada en el vídeo de origen como una marca translúcida. No crea ni activa una máscara de inpainting.", "在源视频的选定区域叠加半透明颜色标记，不会创建或启用修补蒙版。"],
  "🎬 Shot Planner": ["🎬 Planejador de planos", "🎬 Planificador de planos", "🎬 镜头规划器"],
  "— choose a video from the input folder —": ["— escolha um vídeo da pasta de entrada —", "— elige un vídeo de la carpeta de entrada —", "— 从输入文件夹选择视频 —"],
  "⬆ Upload": ["⬆ Enviar", "⬆ Subir", "⬆ 上传"], "Pick or upload a video to start": ["Escolha ou envie um vídeo para começar", "Elige o sube un vídeo para empezar", "选择或上传视频以开始"],
  "It is split into shots the model can generate; every shot then gets its own reference, prompt and mask.": ["O vídeo será dividido em planos que o modelo consegue gerar; cada plano terá sua própria referência, prompt e máscara.", "El vídeo se divide en planos que el modelo puede generar; cada uno tendrá su referencia, prompt y máscara.", "视频会拆分为模型可生成的镜头，每个镜头都有自己的参考图、提示词和蒙版。"],
  "work on a part of the video": ["trabalhe em um trecho do vídeo", "trabaja en una parte del vídeo", "处理视频的一部分"],
  "Start frame": ["Frame inicial", "Fotograma inicial", "起始帧"], "End frame (0 = end)": ["Frame final (0 = fim)", "Fotograma final (0 = final)", "结束帧（0 = 末尾）"],
  "⇤ start here": ["⇤ começar aqui", "⇤ empezar aquí", "⇤ 从此处开始"], "⇥ end here": ["⇥ terminar aqui", "⇥ terminar aquí", "⇥ 在此处结束"], "✕ whole video": ["✕ vídeo inteiro", "✕ vídeo completo", "✕ 整个视频"],
  "Camera cuts or fixed length, on the model's frame grid": ["Cortes de câmera ou duração fixa, na grade de frames do modelo", "Cortes de cámara o duración fija, según la cuadrícula de fotogramas del modelo", "按镜头切换或固定时长，并符合模型帧步长"],
  "re-analyses": ["reanalisar", "vuelve a analizar", "重新分析"], "ignores Max seconds / shot": ["ignora Máx. segundos por plano", "ignora Máx. segundos por plano", "忽略每个镜头的最长秒数"],
  "cropped shots use it for the crop": ["planos recortados usam esse valor no recorte", "los planos recortados usan este valor", "裁剪镜头会使用此尺寸"],
    "Split where the main person changes": ["Dividir quando a pessoa principal mudar", "Dividir cuando cambie la persona principal", "主要人物变化时拆分"],
  "Only run shots with a detected person": ["Executar apenas planos com pessoa detectada", "Ejecutar solo planos con una persona detectada", "仅运行检测到人物的镜头"],
    "adds a boundary when the biggest face switches to someone else (re-splits the plan)": ["cria uma divisão quando o maior rosto muda para outra pessoa (refaz a divisão)", "añade un corte cuando el rostro más grande cambia de persona (vuelve a dividir)", "最大人脸切换到其他人时添加分界（重新拆分计划）"],
  "shots with no detected, non-ignored person are skipped": ["planos sem pessoa detectada e não ignorada são pulados", "se omiten los planos sin una persona detectada y no ignorada", "跳过没有检测到未忽略人物的镜头"],
  "Nobody found yet.": ["Nenhuma pessoa encontrada.", "Aún no se encontraron personas.", "尚未检测到人物。"], "↻ Re-analyse people": ["↻ Analisar pessoas novamente", "↻ Volver a analizar personas", "↻ 重新分析人物"], "👥 Find people": ["👥 Encontrar pessoas", "👥 Buscar personas", "👥 查找人物"],
  "faces sampled every 0.5 s and grouped by identity (InsightFace, CPU)": ["rostos amostrados a cada 0,5 s e agrupados por identidade (InsightFace, CPU)", "rostros muestreados cada 0,5 s y agrupados por identidad (InsightFace, CPU)", "每 0.5 秒采样人脸并按身份分组（InsightFace，CPU）"],
  "detects faces across the video and groups them into people": ["detecta rostos no vídeo e os agrupa por pessoa", "detecta rostros en el vídeo y los agrupa por persona", "检测视频中的人脸并按人物分组"],
  "track face identities to filter or split shots": ["rastreie rostos, associe cada pessoa a uma referência e divida os planos por quem aparece", "rastrea rostros, vincula cada persona a una referencia y divide los planos según quién aparece", "跟踪人脸、关联参考图，并按出镜人物拆分镜头"],
  "Each shot picks what it marks (🎯 text or points, or 🎞 its own mask video) and how it is generated (": ["Cada plano define a seleção (🎯 texto, pontos ou 🎞 vídeo de máscara) e o modo de geração (", "Cada plano define la selección (🎯 texto, puntos o 🎞 vídeo de máscara) y el modo de generación (", "每个镜头设置选区（🎯 文本、点或 🎞 蒙版视频）和生成方式（"],
  "🎞 Mask video (whole video)": ["🎞 Vídeo de máscara (vídeo inteiro)", "🎞 Vídeo de máscara (vídeo completo)", "🎞 蒙版视频（整个视频）"],
  "— none: SAM 3 per shot —": ["— nenhum: SAM 3 por plano —", "— ninguno: SAM 3 por plano —", "— 无：每个镜头使用 SAM 3 —"], "Fill holes": ["Preencher buracos", "Rellenar huecos", "填充孔洞"], "Invert": ["Inverter", "Invertir", "反转"],
  "Show masks on shots": ["Mostrar máscaras nos planos", "Mostrar máscaras en los planos", "在镜头卡片上显示蒙版"], "Expand (px)": ["Expandir (px)", "Expandir (px)", "扩展（像素）"],
  "Temporal expand (frames)": ["Expansão temporal (frames)", "Expansión temporal (fotogramas)", "时间扩展（帧）"], "Detection threshold": ["Limiar de detecção", "Umbral de detección", "检测阈值"], "Max objects": ["Máx. objetos", "Máx. objetos", "最大对象数"],
  "Crop padding (%)": ["Margem do recorte (%)", "Margen del recorte (%)", "裁剪边距（%）"], "Feather (px)": ["Suavização da borda (px)", "Suavizado del borde (px)", "羽化（像素）"], "Paste back": ["Colar de volta", "Pegar de vuelta", "贴回方式"],
  "only the mask (feathered)": ["somente a máscara (suavizada)", "solo la máscara (suavizada)", "仅蒙版（羽化边缘）"], "the whole box (feathered)": ["a caixa inteira (suavizada)", "todo el cuadro (suavizado)", "整个框（羽化边缘）"], "Blockify (px, 0 = off)": ["Blocos (px, 0 = desligado)", "Bloques (px, 0 = desactivado)", "块化（像素，0 = 关闭）"],
  "Uses the official sam3.1_multiplex_fp16 checkpoint (models/checkpoints), downloaded from Comfy-Org/sam3.1 the first time.": ["Usa o checkpoint oficial sam3.1_multiplex_fp16 (models/checkpoints), baixado de Comfy-Org/sam3.1 na primeira utilização.", "Usa el checkpoint oficial sam3.1_multiplex_fp16 (models/checkpoints), descargado de Comfy-Org/sam3.1 la primera vez.", "使用官方 sam3.1_multiplex_fp16 检查点（models/checkpoints），首次运行时从 Comfy-Org/sam3.1 下载。"],
  "SAM 3": ["SAM 3", "SAM 3", "SAM 3"], "track face identities to filter or split shots": ["rastreie rostos, associe cada pessoa a uma referência e organize os planos por quem aparece", "rastrea rostros, vincula cada persona a una referencia y organiza los planos según quién aparece", "跟踪人脸、关联参考图，并按出镜人物分配镜头"],
  "Global reference & prompt": ["Referência e prompt globais", "Referencia y prompt globales", "全局参考图和提示词"], "defaults for every shot": ["padrão para todos os planos", "valor predeterminado para cada plano", "所有镜头的默认设置"],
  "who the shot becomes": ["em quem o plano será transformado", "en quién se transforma el plano", "镜头中的人物将替换为谁"], "Prompt for this shot (empty = global prompt). Placeholders: {target} {details} {shot} {setting}": ["Prompt deste plano (vazio = prompt global). Variáveis: {target} {details} {shot} {setting}", "Prompt de este plano (vacío = prompt global). Variables: {target} {details} {shot} {setting}", "此镜头的提示词（留空则使用全局提示词）。占位符：{target} {details} {shot} {setting}"],
  "recent (click = ref, shift+click = ref 2):": ["recentes (clique = ref., Shift+clique = ref. 2):", "recientes (clic = ref., Mayús+clic = ref. 2):", "最近使用（点击 = 参考图，Shift+点击 = 参考图 2）："], "✕ clear": ["✕ limpar", "✕ borrar", "✕ 清除"], "✕ clear all": ["✕ limpar tudo", "✕ borrar todo", "✕ 全部清除"], "Use global": ["Usar global", "Usar global", "使用全局设置"],
  "empty slots use the global references (Prompts & refs tab)": ["campos vazios usam a referência da pessoa (Elenco) ou a global (aba Prompts e referências)", "los campos vacíos usan la referencia de la persona (Reparto) o la global (Prompts y referencias)", "空白项使用人物参考图（角色）或全局参考图（提示词和参考图选项卡）"],
  "Full frame": ["Quadro inteiro", "Fotograma completo", "整帧"], "The whole frame is regenerated. The mask, if any, only feeds {target} / the setting picture.": ["O quadro inteiro é gerado novamente. A máscara, se houver, só fornece {target} / a imagem de cenário.", "Se vuelve a generar todo el fotograma. La máscara, si existe, solo aporta {target} / la imagen del escenario.", "重新生成整帧。蒙版（如有）仅用于提供 {target} / 场景参考图。"],
  "Only the mask is regenerated, on the whole frame: the rest stays exactly as it was. No crop / uncrop.": ["Somente a área da máscara é regenerada; o restante permanece igual. Sem recorte.", "Solo se regenera el área de la máscara; el resto permanece igual. Sin recorte.", "仅重新生成蒙版区域，其余部分保持不变。不进行裁剪。"],
  "A box around the mask is regenerated (more pixels for a small subject); BFS Shot Join pastes it back.": ["A caixa ao redor da máscara é regenerada (mais pixels para pessoas pequenas); o BFS Shot Join cola o resultado de volta.", "Se regenera el cuadro alrededor de la máscara (más píxeles para sujetos pequeños); BFS Shot Join lo pega de vuelta.", "重新生成蒙版周围的区域（为较小主体提供更多像素）；BFS Shot Join 会将其贴回。"],
  "Inside the crop, only the mask is regenerated: the most detail with the background kept.": ["Dentro do recorte, somente a máscara é regenerada: mais detalhes e fundo preservado.", "Dentro del recorte, solo se regenera la máscara: más detalle y fondo conservado.", "仅重新生成裁剪区域内的蒙版：细节更多，同时保留背景。"],
  "The whole frame is generated (the pose follows the guide, the new subject may be bigger); BFS Shot Join pastes only the subject onto the original: background exact.": ["O quadro inteiro é gerado (a pose segue o guia e o novo sujeito pode ser maior); o BFS Shot Join cola apenas o sujeito no original e preserva o fundo.", "Se genera todo el fotograma (la pose sigue la guía y el sujeto puede ser más grande); BFS Shot Join pega solo al sujeto sobre el original y conserva el fondo.", "生成整帧（姿势遵循引导，主体可更大）；BFS Shot Join 仅将主体贴回原画面，背景保持原样。"],
  "🎯 Points…": ["🎯 Pontos…", "🎯 Puntos…", "🎯 点…"], "👁 Preview": ["👁 Prévia", "👁 Vista previa", "👁 预览"], "✕ Clear": ["✕ Limpar", "✕ Borrar", "✕ 清除"],
  "🎞 mask video": ["🎞 vídeo de máscara", "🎞 vídeo de máscara", "🎞 蒙版视频"], "↳ global prompt": ["↳ prompt global", "↳ prompt global", "↳ 全局提示词"], "— no prompt —": ["— sem prompt —", "— sin prompt —", "— 无提示词 —"],
  "▶ Play all": ["▶ Reproduzir todos", "▶ Reproducir todos", "▶ 播放全部"], "■ Stop": ["■ Parar", "■ Detener", "■ 停止"], "loop shot": ["repetir plano", "repetir plano", "循环播放镜头"],
  "Click the timeline to seek. Play all skips shots that will not run.": ["Clique na linha do tempo para buscar. Reproduzir todos ignora planos que não serão executados.", "Haz clic en la línea de tiempo para buscar. Reproducir todos omite los planos que no se ejecutarán.", "点击时间轴进行定位。播放全部会跳过不运行的镜头。"],
  "Reference & prompt": ["Referência e prompt", "Referencia y prompt", "参考图和提示词"], "Who is replaced": ["Quem será substituído", "Quién se reemplaza", "替换对象"], "Continuity": ["Continuidade", "Continuidad", "连续性"],
  "Global reference & prompt": ["Referência e prompt globais", "Referencia y prompt globales", "全局参考图和提示词"], "Describe references ({details})": ["Descrever referências ({details})", "Describir referencias ({details})", "描述参考图（{details}）"],
  "Custom": ["Personalizado", "Personalizado", "自定义"], "short": ["curto", "breve", "简短"], "full body": ["corpo inteiro", "cuerpo entero", "全身"], "head / face": ["cabeça / rosto", "cabeza / rostro", "头部/面部"], "face attributes": ["características do rosto", "rasgos faciales", "面部特征"], "outfit": ["roupa", "ropa", "服装"],
  "Apply suggestions → all": ["Aplicar sugestões → todos", "Aplicar sugerencias → todos", "应用建议 → 全部"], "fit the prompt": ["ajustar ao prompt", "ajustar al prompt", "适配提示词"],
  "Connect a Qwen3-VL (CLIPLoader) to the planner's vlm input. The VLM buttons work after the workflow has run once with it connected (ComfyUI only hands models to nodes when they run). Each shot's suggestion shows in its editor.": ["Conecte um Qwen3-VL (CLIPLoader) à entrada vlm do planner. Os botões do VLM funcionam depois que o workflow executar uma vez com ele conectado. A sugestão de cada plano aparece no editor.", "Conecta un Qwen3-VL (CLIPLoader) a la entrada vlm del planificador. Los botones del VLM funcionan después de ejecutar el flujo una vez con él conectado. La sugerencia de cada plano aparece en su editor.", "将 Qwen3-VL（CLIPLoader）连接到规划器的 vlm 输入。工作流连接后运行一次，VLM 按钮即可使用。每个镜头的建议会显示在其编辑器中。"],
  "look at every shot and suggests the mask, a description and whether to run it": ["analisa cada plano e sugere máscara, descrição e se deve ser executado", "analiza cada plano y sugiere una máscara, una descripción y si debe ejecutarse", "分析每个镜头并建议蒙版、描述及是否运行"],
  "🎯 Track → whole video": ["🎯 Rastrear → vídeo inteiro", "🎯 Seguir → vídeo completo", "🎯 跟踪 → 整个视频"], "Save (this shot)": ["Salvar (este plano)", "Guardar (este plano)", "保存（此镜头）"], "Save → all shots": ["Salvar → todos os planos", "Guardar → todos los planos", "保存 → 所有镜头"], "Cancel": ["Cancelar", "Cancelar", "取消"], "Clear points": ["Limpar pontos", "Borrar puntos", "清除点"], "👁 Segment": ["👁 Segmentar", "👁 Segmentar", "👁 分割"],
  "Apply these points to EVERY shot, each on its frame at the same relative position (a subject that stays in place)": ["Aplicar estes pontos a TODOS os planos, na mesma posição relativa (para um sujeito que não muda de lugar)", "Aplicar estos puntos a TODOS los planos, en la misma posición relativa (si el sujeto no se mueve)", "将这些点应用到每个镜头的相同相对位置（适用于位置不变的主体）"],
  "the range starts at the player's frame": ["o trecho começa no frame atual do player", "el rango empieza en el fotograma actual", "范围从播放器当前帧开始"], "the range ends after the player's frame": ["o trecho termina após o frame atual do player", "el rango termina después del fotograma actual", "范围在播放器当前帧之后结束"],
  "Nothing to fix in the shots that run.": ["Nenhum problema nos planos que serão executados.", "No hay nada que corregir en los planos que se ejecutarán.", "将运行的镜头没有需要修复的问题。"], "all good": ["tudo certo", "todo bien", "一切正常"], "warning(s)": ["aviso(s)", "aviso(s)", "条警告"], "note(s)": ["observação(ões)", "nota(s)", "条备注"],
  "quick tests": ["testes rápidos", "pruebas rápidas", "快速测试"], "from the start of the range": ["a partir do início do trecho", "desde el inicio del rango", "从范围起点开始"], "Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video.": ["Cada execução gera e salva um plano. Os nodes após BFS Shot Join só executam no último plano, com o vídeo completo.", "Cada ejecución genera y guarda un plano. Los nodos posteriores a BFS Shot Join solo se ejecutan en el último plano, con el vídeo completo.", "每次运行生成并保存一个镜头。BFS Shot Join 后的节点仅在最后一个镜头运行，并接收完整视频。"],
  "skipped shots do not run": ["planos ignorados não são executados", "los planos omitidos no se ejecutan", "跳过的镜头不会运行"], "only shots that run": ["somente planos executados", "solo los planos que se ejecutan", "仅统计将运行的镜头"],
  "off": ["desligado", "desactivado", "关闭"], "on": ["ligado", "activado", "开启"], "run": ["executar", "ejecutar", "运行"], "skip": ["ignorar", "omitir", "跳过"], "filters decide": ["filtros decidem", "deciden los filtros", "由筛选条件决定"], "always run": ["sempre executar", "ejecutar siempre", "始终运行"], "always skip": ["sempre ignorar", "omitir siempre", "始终跳过"], "⏸ Disable": ["⏸ Desativar", "⏸ Desactivar", "⏸ 禁用"], "▶ Enable": ["▶ Ativar", "▶ Activar", "▶ 启用"], "✂ Split": ["✂ Dividir", "✂ Dividir", "✂ 拆分"], "⇥ Merge": ["⇥ Unir", "⇥ Unir", "⇥ 合并"],
  "Advanced": ["Avançado", "Avanzado", "高级"], "English": ["Inglês", "Inglés", "英语"], "Português": ["Português", "Portugués", "葡萄牙语"], "Español": ["Espanhol", "Español", "西班牙语"], "中文": ["Chinês", "Chino", "中文"],
  "Video": ["Vídeo", "Vídeo", "视频"], "Shots": ["Planos", "Planos", "镜头"], "Prompts": ["Prompts", "Prompts", "提示词"],
  "Marker color · Ref 1": ["Cor do marcador · Ref. 1", "Color de marca · Ref. 1", "标记颜色 · 参考图 1"], "Marker color · Ref 2": ["Cor do marcador · Ref. 2", "Color de marca · Ref. 2", "标记颜色 · 参考图 2"],
  "These are the default marker colors for the global references. Turn on Colored Mark on a shot to apply them; shot-level colors can override them.": ["Estas são as cores padrão das referências globais. Ative Marcação colorida no plano para usá-las; cada plano pode substituí-las.", "Estos son los colores predeterminados de las referencias globales. Activa Marca de color en un plano para aplicarlos; cada plano puede cambiarlos.", "这些是全局参考图的默认标记颜色。启用镜头的彩色标记即可应用；镜头也可以单独覆盖。"],
  "Audio": ["Áudio", "Audio", "音频"], "use the video's audio": ["usar o áudio do vídeo", "usar el audio del vídeo", "使用视频音频"], "silent track": ["faixa silenciosa", "pista silenciosa", "静音音轨"],
  "auto": ["automático", "automático", "自动"], "fixed": ["fixo", "fijo", "固定"], "single": ["único", "único", "单个"], "adaptive": ["adaptativo", "adaptativo", "自适应"], "content": ["conteúdo", "contenido", "内容"], "builtin": ["integrado", "integrado", "内置"],
  "PySceneDetect adaptive": ["PySceneDetect adaptativo", "PySceneDetect adaptativo", "PySceneDetect 自适应"], "PySceneDetect content": ["PySceneDetect por conteúdo", "PySceneDetect por contenido", "PySceneDetect 内容检测"], "Built-in": ["Integrado", "Integrado", "内置"],
  "Auto loop": ["Loop automático", "Bucle automático", "自动循环"], "Queue loop": ["Loop em fila", "Bucle en cola", "队列循环"], "queue loop": ["loop em fila", "bucle en cola", "队列循环"], "auto loop": ["loop automático", "bucle automático", "自动循环"],
  "no audio": ["sem áudio", "sin audio", "无音频"], "silent track": ["faixa silenciosa", "pista silenciosa", "静音音轨"], "timeline": ["linha do tempo", "línea de tiempo", "时间轴"], "generate at": ["gerar em", "generar a", "生成尺寸"],
  "Max person size (0-1 of frame)": ["Tamanho mínimo da pessoa (0–1 do quadro)", "Tamaño mínimo de persona (0–1 del fotograma)", "人物最小尺寸（占画面 0–1）"], "Min person size (0-1 of frame)": ["Tamanho mínimo da pessoa (0–1 do quadro)", "Tamaño mínimo de persona (0–1 del fotograma)", "人物最小尺寸（占画面 0–1）"],
  "Max people (0 = any)": ["Máx. pessoas (0 = qualquer)", "Máx. personas (0 = cualquiera)", "人物上限（0 = 不限）"], "Min frames (0 = off)": ["Mín. frames (0 = desligado)", "Mín. fotogramas (0 = desactivado)", "最少帧数（0 = 关闭）"], "Frames sampled / shot": ["Frames amostrados por plano", "Fotogramas muestreados por plano", "每个镜头采样帧数"], "Dark below (0-1)": ["Escuro abaixo de (0–1)", "Oscuro por debajo de (0–1)", "暗度低于（0–1）"], "Static below": ["Estático abaixo de", "Estático por debajo de", "静止度低于"],
  "e.g. 0.03 skips wide shots": ["ex.: 0,03 ignora planos abertos", "p. ej., 0,03 omite planos amplios", "例如 0.03 会跳过远景镜头"], "skip crowds": ["ignorar multidões", "omitir multitudes", "跳过人群"], "re-analyses": ["reanalisar", "vuelve a analizar", "重新分析"],
  "Describe target": ["Descrever alvo", "Describir objetivo", "描述目标"], "🧑 Describe target": ["🧑 Descrever alvo", "🧑 Describir objetivo", "🧑 描述目标"], "new subject": ["novo sujeito", "nuevo sujeto", "新主体"], "person (or dog, cat, car… what the result shows)": ["pessoa (ou cão, gato, carro… o que aparecerá no resultado)", "persona (o perro, gato, coche… lo que aparecerá en el resultado)", "人物（或狗、猫、汽车等结果中的主体）"],
  "The second person (optional)": ["Segunda pessoa (opcional)", "Segunda persona (opcional)", "第二个人（可选）"], "Follow cast": ["Seguir identidade", "Seguir reparto", "跟随角色"], "manual identity": ["identidade manual", "identidad manual", "手动身份"],
  "Guide video": ["Vídeo guia", "Vídeo guía", "引导视频"], "Reference": ["Referência", "Referencia", "参考图"], "Prompt": ["Prompt", "Prompt", "提示词"], "reference": ["referência", "referencia", "参考图"],
  "on": ["ativado", "activado", "开启"], "off": ["desativado", "desactivado", "关闭"], "not analysed": ["não analisado", "sin analizar", "未分析"], "not analyzed": ["não analisado", "sin analizar", "未分析"],
  "No shots yet": ["Nenhum plano ainda", "Aún no hay planos", "尚无镜头"], "No video": ["Nenhum vídeo", "Sin vídeo", "没有视频"], "No faces": ["Nenhum rosto", "Sin rostros", "没有人脸"], "no faces": ["sem rostos", "sin rostros", "没有人脸"],
  "cut": ["corte", "corte", "切换"], "run": ["executar", "ejecutar", "运行"], "skip": ["ignorar", "omitir", "跳过"], "skip this shot": ["ignorar este plano", "omitir este plano", "跳过此镜头"],
  "people": ["pessoas", "personas", "人物"], "person": ["pessoa", "persona", "人物"], "shots": ["planos", "planos", "镜头"], "shot(s)": ["plano(s)", "plano(s)", "个镜头"], "linked": ["vinculada", "vinculada", "已关联"], "main": ["principal", "principal", "主要"],
  "No VLM yet": ["Nenhum VLM carregado", "Aún no hay un VLM", "尚未加载 VLM"], "Connect a Qwen3-VL": ["Conecte um Qwen3-VL", "Conecta un Qwen3-VL", "连接 Qwen3-VL"],
  "↻ Status": ["↻ Status", "↻ Estado", "↻ 状态"], "⟲ Reset loop": ["⟲ Reiniciar loop", "⟲ Reiniciar bucle", "⟲ 重置循环"], "Queue loop": ["Loop em fila", "Bucle en cola", "队列循环"],
  "Start here": ["Começar aqui", "Empezar aquí", "从此处开始"], "End here": ["Terminar aqui", "Terminar aquí", "在此处结束"], "reset": ["redefinir", "restablecer", "重置"], "all shots": ["todos os planos", "todos los planos", "所有镜头"], "this shot": ["este plano", "este plano", "此镜头"],
  "Describe target": ["Descrever alvo", "Describir objetivo", "描述目标"], "Settings": ["Configurações", "Ajustes", "设置"], "mask text": ["texto da máscara", "texto de máscara", "蒙版描述"],
  "The global reference and prompt are used when a shot has no values of its own.": ["A referência e o prompt globais são usados quando um plano não tem valores próprios.", "La referencia y el prompt globales se usan cuando un plano no tiene valores propios.", "镜头没有单独设置时会使用全局参考图和提示词。"],
  "A VLM is optional": ["O VLM é opcional", "El VLM es opcional", "VLM 为可选项"], "Optional": ["Opcional", "Opcional", "可选"], "optional": ["opcional", "opcional", "可选"],
  "person in white": ["pessoa de branco", "persona de blanco", "穿白衣的人"], "woman in pink top": ["mulher de blusa rosa", "mujer con blusa rosa", "穿粉色上衣的女人"],
  "nothing found": ["nada encontrado", "no se encontró nada", "未找到内容"], "not found": ["não encontrado", "no encontrado", "未找到"], "Preview": ["Prévia", "Vista previa", "预览"],
  "Mask only / Crop + mask": ["Somente máscara / Recorte + máscara", "Solo máscara / Recorte + máscara", "仅蒙版 / 裁剪并蒙版"], "per shot (planner)": ["por plano (planner)", "por plano (planificador)", "逐镜头（规划器）"],
  "Person": ["Pessoa", "Persona", "人物"], "shot": ["plano", "plano", "镜头"], "shot(s)": ["plano(s)", "plano(s)", "个镜头"],
  "mask covers": ["máscara cobre", "la máscara cubre", "蒙版覆盖"], "nothing to fix": ["nada a corrigir", "nada que corregir", "无需修复"],
  "Camera cuts: shots start at the detected cuts, long ones are split evenly. Fixed length: equal parts no longer than the maximum.": ["Cortes de câmera: os planos começam nos cortes detectados; planos longos são divididos igualmente. Duração fixa: partes iguais limitadas pelo máximo.", "Cortes de cámara: los planos empiezan en los cortes detectados; los largos se dividen por igual. Duración fija: partes iguales con el máximo indicado.", "镜头切换：镜头从检测到的切点开始，较长镜头会均匀拆分。固定时长：拆分为不超过上限的等长片段。"],
  "Cut detector. PySceneDetect adaptive works on most footage (fast camera moves don't trigger it); content is stricter on hard cuts; built-in needs no package.": ["Detector de cortes. O modo adaptativo do PySceneDetect funciona na maioria dos vídeos (movimentos rápidos de câmera não acionam cortes); o modo conteúdo é mais rigoroso; o integrado não exige pacote.", "Detector de cortes. El modo adaptativo de PySceneDetect funciona en la mayoría de vídeos (los movimientos rápidos no activan cortes); contenido es más estricto; el integrado no requiere paquetes.", "镜头切换检测器。PySceneDetect 自适应模式适用于大多数素材（快速运镜不会触发切点）；内容模式对硬切更严格；内置模式无需额外软件包。"],
  "How easily a change counts as a cut. Higher finds more cuts (and more false ones).": ["O quanto uma mudança é considerada corte. Um valor maior encontra mais cortes e também mais falsos positivos.", "Facilidad con la que un cambio cuenta como corte. Un valor mayor detecta más cortes y también más falsos positivos.", "变化被判定为切点的难易程度。数值越高，检测到的切点越多，误报也越多。"],
  "Frame counts the model accepts. H3: 17n+5, LTX/Wan: 8n+1, Wan: 4n+1. Each shot is generated at the next valid length and trimmed back.": ["Contagens de frames aceitas pelo modelo. H3: 17n+5; LTX/Wan: 8n+1; Wan: 4n+1. Cada plano é gerado no próximo comprimento válido e depois aparado.", "Cantidad de fotogramas que acepta el modelo. H3: 17n+5; LTX/Wan: 8n+1; Wan: 4n+1. Cada plano se genera con la siguiente duración válida y luego se recorta.", "模型接受的帧数规则。H3：17n+5；LTX/Wan：8n+1；Wan：4n+1。每个镜头按下一个有效长度生成后再裁短。"],
  "Longest shot sent to the model. Keep it within what the model / LoRA was trained on (H3 body swap: about 4.5 s).": ["Duração máxima enviada ao modelo. Mantenha dentro do que o modelo/LoRA aprendeu (body swap H3: cerca de 4,5 s).", "Duración máxima enviada al modelo. Respeta lo usado en el entrenamiento del modelo/LoRA (body swap H3: unos 4,5 s).", "发送给模型的镜头最长时长。应符合模型/LoRA 的训练长度（H3 换人约 4.5 秒）。"],
  "Shorter shots merge into a neighbour when the merge still fits the maximum.": ["Planos mais curtos são unidos ao vizinho quando a junção ainda respeita o máximo.", "Los planos más cortos se unen al vecino si la unión respeta el máximo.", "较短镜头会在合并后仍未超过上限时与相邻镜头合并。"],
  "The planner works on this frame rate (the source is resampled). Generated video and audio use it too.": ["O planner usa esta taxa de frames (o vídeo de origem é reamostrado). O vídeo e áudio gerados também a usam.", "El planificador usa esta frecuencia de fotogramas (el origen se remuestrea). El vídeo y el audio generados también la usan.", "规划器使用此帧率（源视频会重新采样）。生成的视频和音频也会使用此帧率。"],
  "Only the first N shots (0 = all). Handy for quick tests.": ["Somente os primeiros N planos (0 = todos). Útil para testes rápidos.", "Solo los primeros N planos (0 = todos). Útil para pruebas rápidas.", "仅处理前 N 个镜头（0 = 全部）。适合快速测试。"],
  "Only the first N seconds of the video (0 = all).": ["Somente os primeiros N segundos do vídeo (0 = todos).", "Solo los primeros N segundos del vídeo (0 = todos).", "仅处理视频前 N 秒（0 = 全部）。"],
  "Generation size: the source aspect ratio at this pixel area (0.15 ≈ 512×288). Cropped shots use it for the crop.": ["Tamanho da geração: mantém a proporção do vídeo nessa área de pixels (0,15 ≈ 512×288). Planos recortados usam esse valor no recorte.", "Tamaño de generación: mantiene la proporción de origen con esta área de píxeles (0,15 ≈ 512×288). Los planos recortados usan este valor.", "生成尺寸：按此像素面积保持源视频宽高比（0.15 ≈ 512×288）。裁剪镜头使用此尺寸。"],
  "Width and height snap to this multiple (32 for H3 and most video models).": ["Largura e altura são ajustadas a este múltiplo (32 para H3 e a maioria dos modelos de vídeo).", "El ancho y alto se ajustan a este múltiplo (32 para H3 y la mayoría de modelos de vídeo).", "宽度和高度会对齐到此倍数（H3 和多数视频模型使用 32）。"],
  "Auto loop: every shot in one queue run. Queue loop: one shot per run, stored on disk, the join outputs the video after the last shot.": ["Loop automático: todos os planos em uma execução da fila. Loop em fila: um plano por execução, salvo em disco; o join gera o vídeo após o último plano.", "Bucle automático: todos los planos en una ejecución. Bucle en cola: un plano por ejecución, guardado en disco; el join genera el vídeo al terminar el último.", "自动循环：一次队列运行所有镜头。队列循环：每次运行一个镜头并保存到磁盘；最后一个镜头完成后由 Join 输出视频。"],
  "Shots that do not run: keep the original video there, or remove them (with their audio).": ["Planos não executados: manter o vídeo original nesses trechos ou removê-los junto com o áudio.", "Planos que no se ejecutan: conserva el vídeo original en ese tramo o elimínalo junto con su audio.", "未运行的镜头：保留对应原视频片段，或连同音频一起移除。"],
  "Context kept around the mask's box, as a % of the box. More context blends better; less gives the model more pixels on the subject.": ["Contexto mantido ao redor da caixa da máscara, como porcentagem da caixa. Mais contexto mistura melhor; menos dá mais pixels ao sujeito.", "Contexto alrededor del cuadro de la máscara, como porcentaje. Más contexto mezcla mejor; menos ofrece más píxeles al sujeto.", "蒙版边框周围保留的上下文比例。上下文越多越容易融合；越少则主体像素越多。"],
  "Grow the mask by this many pixels: before pasting a cropped result back, and for the generation mask (inpaint 'only the mask'). Raise it when the new person is bigger than the old one (longer hair, wider body).": ["Expande a máscara por esta quantidade de pixels antes de colar o recorte e na máscara de geração. Aumente se a nova pessoa for maior (cabelo mais comprido, corpo mais largo).", "Amplía la máscara esta cantidad de píxeles antes de pegar el recorte y para la máscara de generación. Auméntala si la persona nueva es más grande.", "粘贴裁剪结果前以及生成蒙版时，将蒙版扩展指定像素。新人物更大时（头发更长、身体更宽）可调高。"],
  "Soft edge of the paste-back, in pixels.": ["Suavização da borda ao colar de volta, em pixels.", "Suavizado del borde al pegar de vuelta, en píxeles.", "贴回时边缘的柔化像素数。"],
  "Hold the mask this many frames before and after: removes flicker where SAM 3 drops a frame.": ["Estende a máscara por esta quantidade de frames antes e depois; reduz falhas quando o SAM 3 perde um frame.", "Mantiene la máscara estos fotogramas antes y después; reduce parpadeos cuando SAM 3 pierde uno.", "蒙版前后延续指定帧数，减少 SAM 3 漏掉某帧时的闪烁。"],
  "Square blocks of this size (0 = off). 16 matches H3's latent grid, useful as an inpainting mask.": ["Blocos quadrados deste tamanho (0 = desligado). 16 corresponde à grade latente do H3 e pode ser útil como máscara de inpainting.", "Bloques cuadrados de este tamaño (0 = desactivado). 16 coincide con la cuadrícula latente de H3 y sirve para inpainting.", "使用此尺寸的方块（0 = 关闭）。16 与 H3 潜空间网格匹配，可用于修补蒙版。"],
  "SAM 3 score needed to keep a text match. Lower finds more (and more wrong) objects.": ["Pontuação mínima do SAM 3 para manter uma correspondência por texto. Valores menores encontram mais objetos, inclusive incorretos.", "Puntuación mínima de SAM 3 para conservar una coincidencia de texto. Un valor menor encuentra más objetos, también erróneos.", "保留文本匹配所需的 SAM 3 分数。调低会找到更多对象，也会增加误检。"],
  "How many matches of the text prompt are tracked together.": ["Quantas correspondências do texto serão rastreadas juntas.", "Cuántas coincidencias del texto se siguen a la vez.", "同时跟踪多少个符合文本的对象。"],
  "Only the mask: the result replaces the masked pixels. The whole box: the full crop is pasted with feathered borders.": ["Somente máscara: o resultado substitui os pixels mascarados. Caixa inteira: todo o recorte é colado com bordas suaves.", "Solo máscara: el resultado sustituye los píxeles enmascarados. Cuadro completo: se pega todo el recorte con bordes suavizados.", "仅蒙版：结果替换蒙版像素。整个框：粘贴完整裁剪区域并柔化边缘。"],
  "How many frames of each shot the VLM looks at.": ["Quantos frames de cada plano o VLM analisa.", "Cuántos fotogramas de cada plano analiza el VLM.", "VLM 分析每个镜头的帧数。"], "Longest VLM answer. Raise it for long descriptions.": ["Tamanho máximo da resposta do VLM. Aumente para descrições longas.", "Longitud máxima de respuesta del VLM. Auméntala para descripciones largas.", "VLM 回复的最大长度。长描述可调高。"],
  "skipped by hand": ["ignorado manualmente", "omitido manualmente", "手动跳过"], "no detected person": ["nenhuma pessoa detectada", "no se detectó ninguna persona", "未检测到人物"],
  "longer than {…} frames: split it": ["mais longo que {…} frames: divida o plano", "supera {…} fotogramas: divide el plano", "超过 {…} 帧：请拆分镜头"],
  "{…} needs a mask (text, points or a mask video; or the planner's mask input)": ["{…} precisa de uma máscara (texto, pontos, vídeo de máscara ou entrada de máscara do Planner)", "{…} necesita una máscara (texto, puntos, vídeo de máscara o entrada de máscara del planificador)", "{…} 需要蒙版（文本、点、蒙版视频或 Planner 的蒙版输入）"],
  "the mask preview found nothing": ["a prévia da máscara não encontrou nada", "la vista previa de la máscara no encontró nada", "蒙版预览未找到目标"],
  "the prompt uses {target} but the shot has no target description": ["o prompt usa {target}, mas o plano não tem uma descrição do alvo", "el prompt usa {target}, pero el plano no tiene una descripción del objetivo", "提示词使用了 {target}，但镜头没有目标描述"],
  "the prompt uses {details} but its references have no description yet": ["o prompt usa {details}, mas as referências ainda não têm descrição", "el prompt usa {details}, pero las referencias aún no tienen descripción", "提示词使用了 {details}，但参考图还没有描述"],
  "{details} sits inside a sentence of the prompt: use the 'short' describe preset (Prompts & refs), the other presets write a paragraph": ["{details} está dentro de uma frase do prompt: use a descrição 'short' (Prompts e referências); os outros modos escrevem um parágrafo", "{details} está dentro de una frase del prompt: usa la opción 'short' (Prompts y referencias); las demás opciones escriben un párrafo", "{details} 位于提示词句子中：请使用“short”描述模式（Prompts & refs），其他模式会生成段落"],
  "no prompt (fine when the node's prompt input is connected)": ["sem prompt (tudo certo se a entrada prompt do node estiver conectada)", "sin prompt (no pasa nada si la entrada prompt del nodo está conectada)", "没有提示词（若节点的 prompt 输入已连接则无需处理）"],
  "no reference (fine when the node's ref_image input is connected)": ["sem referência (tudo certo se a entrada ref_image do node estiver conectada)", "sin referencia (no pasa nada si la entrada ref_image del nodo está conectada)", "没有参考图（若节点的 ref_image 输入已连接则无需处理）"],
  "Analysing video…": ["Analisando o vídeo…", "Analizando el vídeo…", "正在分析视频…"], "Detecting cuts…": ["Detectando cortes…", "Detectando cortes…", "正在检测切点…"], "Detecting people and faces…": ["Detectando pessoas e rostos…", "Detectando personas y rostros…", "正在检测人物和人脸…"],
  "Finding people by face…": ["Localizando pessoas pelos rostos…", "Buscando personas por sus rostros…", "正在通过人脸查找人物…"], "Selecting the person (SAM 3) and describing them (VLM)…": ["Selecionando a pessoa (SAM 3) e descrevendo-a (VLM)…", "Seleccionando a la persona (SAM 3) y describiéndola (VLM)…", "正在选择人物（SAM 3）并生成描述（VLM）…"],
  "The SAM 3 target cutout was created (shown below), but no VLM is loaded to describe it. Connect Qwen3-VL to the Shot Planner's vlm input and queue the workflow once; then click Describe target again. You can also type the target description yourself.": ["O recorte do alvo pelo SAM 3 foi criado abaixo, mas nenhum VLM está carregado para descrevê-lo. Conecte o Qwen3-VL à entrada vlm do Shot Planner e execute a fila uma vez; depois clique novamente em Descrever alvo. Você também pode escrever a descrição manualmente.", "SAM 3 creó el recorte del objetivo (se muestra abajo), pero no hay ningún VLM cargado para describirlo. Conecta Qwen3-VL a la entrada vlm de Shot Planner y ejecuta el flujo una vez; después vuelve a pulsar Describir objetivo. También puedes escribir la descripción.", "SAM 3 已生成目标抠图（见下方），但尚未加载 VLM 来描述它。将 Qwen3-VL 连接到 Shot Planner 的 vlm 输入并运行一次队列，然后再次点击“描述目标”。也可以手动填写描述。"],
  "Segmenting with SAM 3… (the first time downloads/loads the model)": ["Segmentando com SAM 3… (na primeira vez, o modelo será baixado/carregado)", "Segmentando con SAM 3… (la primera vez se descarga/carga el modelo)", "正在使用 SAM 3 分割…（首次运行会下载并加载模型）"],
  "open the checks (Run tab)": ["abrir as verificações (aba Executar)", "abrir las comprobaciones (pestaña Ejecutar)", "打开检查项（Run 标签页）"], "Open the full guide (every setting explained, with examples)": ["Abrir o guia completo (com explicações e exemplos de cada opção)", "Abrir la guía completa (explicaciones y ejemplos de cada opción)", "打开完整指南（包含各设置的说明和示例）"],
  "click = keep (green) · right-click or shift+click = exclude (red) · click a point to remove it": ["clique = manter (verde) · clique direito ou Shift+clique = excluir (vermelho) · clique em um ponto para removê-lo", "clic = conservar (verde) · clic derecho o Mayús+clic = excluir (rojo) · clic en un punto para quitarlo", "单击 = 保留（绿色）· 右键或 Shift+单击 = 排除（红色）· 单击已有点可将其删除"],
  "click to remove": ["clique para remover", "haz clic para quitar", "点击删除"], "text prompt (used when there are no points)": ["texto de seleção (usado quando não há pontos)", "texto de selección (se usa cuando no hay puntos)", "文本提示（没有标记点时使用）"],
  "Save → all shots": ["Salvar → todos os planos", "Guardar → todos los planos", "保存 → 所有镜头"], "Save (this shot)": ["Salvar (este plano)", "Guardar (este plano)", "保存（此镜头）"],
  "Track this selection with SAM 3 through the whole video (or the range), forward and backward from this frame; every shot without its own selection takes its part of the track. Also type a short text (e.g. 'woman in pink top') so the subject is found again after camera cuts.": ["Rastreie esta seleção com SAM 3 pelo vídeo inteiro (ou trecho), avançando e retrocedendo a partir deste frame. Cada plano sem seleção própria usa sua parte do rastreamento. Digite também uma descrição curta (por exemplo, 'mulher de blusa rosa') para reencontrar o sujeito após os cortes.", "Sigue esta selección con SAM 3 por todo el vídeo (o tramo), hacia delante y atrás desde este fotograma. Cada plano sin selección propia usa su parte del seguimiento. Añade también una descripción breve (por ejemplo, 'mujer con blusa rosa') para volver a encontrar al sujeto después de los cortes.", "使用 SAM 3 从此帧开始向前和向后跟踪所选对象，覆盖整个视频（或片段）。没有单独选区的镜头会使用对应的跟踪结果。再输入简短描述（如“穿粉色上衣的女性”），以便在切镜后重新找到主体。"],
  "Colored Mark adds a translucent color overlay to a selected region in the source video. It does not enable inpainting. Choose the color before marking the region with points or a description.": ["Marcação colorida aplica uma sobreposição translúcida à região selecionada no vídeo de origem. Ela não ativa inpainting. Escolha a cor antes de marcar a região com pontos ou descrição.", "Marca de color aplica una superposición translúcida a la región seleccionada del vídeo original. No activa el inpainting. Elige el color antes de marcar con puntos o una descripción.", "彩色标记会在源视频的选定区域上叠加半透明颜色，不会启用修补。请先选颜色，再用点或描述标记区域。"],
  "choose which shots run": ["escolher quais planos serão executados", "elegir qué planos se ejecutan", "选择要运行的镜头"],
  "Fill enclosed gaps so each region is solid": ["Preencher espaços fechados para deixar cada região sólida", "Rellenar huecos cerrados para que cada región sea sólida", "填充封闭空隙，使区域完整"], "Use everything except the segmented object": ["Usar tudo, exceto o objeto segmentado", "Usar todo excepto el objeto segmentado", "使用除已分割对象以外的所有区域"], "Overlay the mask preview on the shot cards": ["Sobrepor a prévia da máscara nos cartões dos planos", "Superponer la vista previa de la máscara en las tarjetas de los planos", "在镜头卡片上叠加蒙版预览"],
  "runs the detectors on a few frames of every shot": ["executa os detectores em alguns frames de cada plano", "ejecuta los detectores en algunos fotogramas de cada plano", "在每个镜头的若干帧上运行检测器"],
  "Global prompt (a connected `prompt` input overrides it)": ["Prompt global (a entrada `prompt` conectada tem prioridade)", "Prompt global (la entrada `prompt` conectada tiene prioridad)", "全局提示词（若连接了 `prompt` 输入，则以输入为准）"],
  "Used by shots without their own. Connected ref_image / ref_image_2 / prompt inputs on the node override these. Placeholders:": ["Usado nos planos sem valores próprios. As entradas ref_image / ref_image_2 / prompt conectadas ao node têm prioridade. Marcadores:", "Se usa en los planos sin valores propios. Las entradas ref_image / ref_image_2 / prompt conectadas al nodo tienen prioridad. Marcadores:", "镜头没有单独设置时使用。节点上连接的 ref_image / ref_image_2 / prompt 输入优先。占位符："],
  "description of the shot's references ·": ["descrição das referências do plano ·", "descripción de las referencias del plano ·", "镜头参考图描述 ·"], "who is replaced (per shot) ·": ["quem será substituído (por plano) ·", "quién se sustituye (por plano) ·", "被替换对象（逐镜头）·"], "the VLM's description of the shot ·": ["descrição do plano pelo VLM ·", "descripción del plano generada por el VLM ·", "VLM 对镜头的描述 ·"], "the setting picture (H3 Conditioning).": ["imagem de cenário (H3 Conditioning)。", "imagen del entorno (H3 Conditioning).", "场景参考图（H3 Conditioning）。"],
  "At run time the planner asks the VLM about the shots it runs and applies the options below": ["Durante a execução, o Planner consulta o VLM sobre os planos processados e aplica as opções abaixo", "Durante la ejecución, el planificador consulta al VLM sobre los planos y aplica las opciones siguientes", "运行时，Planner 会让 VLM 分析要处理的镜头并应用以下选项"],
  "Shots without a mask text or points get the VLM's segment suggestion": ["Planos sem texto de máscara ou pontos recebem uma sugestão de segmentação do VLM", "Los planos sin texto de máscara ni puntos reciben una sugerencia de segmentación del VLM", "没有蒙版文本或标记点的镜头会使用 VLM 的分割建议"],
  "Write {shot} in a prompt: it becomes the VLM's description of that shot (camera, framing, action)": ["Use {shot} no prompt para inserir a descrição do plano pelo VLM (câmera, enquadramento e ação)", "Escribe {shot} en el prompt para insertar la descripción del plano generada por el VLM (cámara, encuadre y acción)", "在提示词中写入 {shot}，即可插入 VLM 对镜头的描述（摄影机、构图和动作）"],
  "Mask text for shots without one, skip where the VLM says skip": ["Texto de máscara para planos sem seleção; ignora os que o VLM indicar", "Texto de máscara para planos sin selección; omite los que el VLM marque", "为未设置选区的镜头填入蒙版文本，并跳过 VLM 建议跳过的镜头"],
  "looks at every shot and suggests the mask, a description and whether to run it": ["analisa cada plano e sugere a máscara, uma descrição e se deve executá-lo", "analiza cada plano y sugiere la máscara, una descripción y si debe ejecutarse", "分析每个镜头并建议蒙版、描述以及是否运行"],
  "Try another description: a new seed (and some temperature if it is 0), then describe again": ["Tentar outra descrição: usar uma nova seed (e aumentar a temperatura se estiver em 0) e descrever novamente", "Probar otra descripción: usar una nueva seed (y subir la temperatura si está en 0) y volver a describir", "重新生成描述：更换 seed（若温度为 0 则适当提高），再分析一次"],
  "0 = always the same answer (greedy); 0.5-1 = varied answers, picked by the seed": ["0 = resposta sempre igual (greedy); 0,5–1 = respostas variadas, controladas pela seed", "0 = respuesta siempre igual (greedy); 0,5–1 = respuestas variadas según la seed", "0 = 始终生成相同答案（贪心）；0.5–1 = 由 seed 控制的多样答案"],
  "Used when the temperature is above 0: another seed, another description": ["Usada quando a temperatura é maior que 0: outra seed gera outra descrição", "Se usa cuando la temperatura es mayor que 0: otra seed genera otra descripción", "温度大于 0 时使用：更换 seed 可生成不同描述"],
  "The VLM gets the global prompt's sentence that holds {details} (and the shot prompt's {target} sentence for 🧑 Describe target) with a blank in its place, and writes the description to fit it: the description adapts to the prompt, not the other way round.": ["O VLM recebe a frase do prompt global que contém {details} (ou a frase com {target} ao usar 🧑 Descrever alvo), com o marcador em branco, e escreve uma descrição que se encaixa nela.", "El VLM recibe la frase del prompt global que contiene {details} (o la frase con {target} al usar 🧑 Describir objetivo), con el marcador en blanco, y escribe una descripción que encaje.", "VLM 会读取包含 {details} 的全局提示词句子（使用 🧑 描述目标时则读取含 {target} 的镜头提示词），并在占位处生成适配的描述。"],
  "Add a reference (global, per shot or per shot) first.": ["Adicione uma referência global ou por plano primeiro.", "Añade primero una referencia global o por plano.", "请先添加全局参考图或镜头参考图。"],
  "✓ Nothing to fix in the shots that run.": ["✓ Nenhum ajuste necessário nos planos que serão executados.", "✓ No hay nada que corregir en los planos que se ejecutarán.", "✓ 将运行的镜头无需修正。"],
  "Keep original video": ["Manter o vídeo original", "Conservar el vídeo original", "保留原视频"],
  "Each run generates one shot and stores it. Nodes after BFS Shot Join only run on the last shot, with the full video.": ["Cada execução gera e salva um plano. Os nodes após BFS Shot Join só executam no último, com o vídeo completo.", "Cada ejecución genera y guarda un plano. Los nodos posteriores a BFS Shot Join solo se ejecutan en el último, con el vídeo completo.", "每次运行生成并保存一个镜头。BFS Shot Join 后的节点仅在最后一个镜头运行，并接收完整视频。"],
  "go to this shot": ["ir para este plano", "ir a este plano", "跳转到此镜头"], "Pick a video in the Video tab.": ["Escolha um vídeo na aba Vídeo.", "Elige un vídeo en la pestaña Vídeo.", "请在 Video 标签页中选择视频。"],
  "split this shot in two": ["dividir este plano em dois", "dividir este plano en dos", "将此镜头一分为二"], "merge with the next shot": ["unir ao próximo plano", "unir con el siguiente plano", "与下一个镜头合并"],
  "Forget the recent pictures (the ones this plan still uses stay)": ["Esquecer as imagens recentes (as usadas neste plano serão mantidas)", "Olvidar las imágenes recientes (se conservan las que usa este plan)", "清除最近使用的图片（本计划仍在使用的图片会保留）"],
  "Remove EVERY reference from this plan (global and all shots) and the recent list": ["Remover TODAS as referências deste plano (global e de todos os planos) e da lista recente", "Eliminar TODAS las referencias de este plan (globales y de todos los planos) y de la lista reciente", "从此计划（全局和所有镜头）及最近列表中移除所有参考图"],
  "clear this shot's references and prompt: it uses the global ones": ["limpar as referências e o prompt deste plano para usar os globais", "borrar las referencias y el prompt de este plano para usar los globales", "清除此镜头的参考图和提示词，改用全局设置"],
  "+ Mark a second region (optional)": ["+ Marcar uma segunda região (opcional)", "+ Marcar una segunda región (opcional)", "+ 标记第二个区域（可选）"], "Mark a second region · Ref 2": ["Marcar segunda região · Ref. 2", "Marcar segunda región · Ref. 2", "标记第二个区域 · 参考图 2"], "Describe second region": ["Descrever segunda região", "Describir la segunda región", "描述第二个区域"], "e.g. the person on the left": ["ex.: a pessoa à esquerda", "p. ej., la persona de la izquierda", "例如：左侧的人物"],
  "mask: what to segment, in English (woman in pink top, black dog…) · Enter = preview": ["máscara: o que segmentar, em inglês (woman in pink top, black dog…) · Enter = prévia", "máscara: qué segmentar, en inglés (woman in pink top, black dog…) · Enter = vista previa", "蒙版：用英文描述分割对象（woman in pink top、black dog…）· Enter = 预览"],
  "Pick positive / negative points on a frame of this shot (more precise than text when several people look alike)": ["Marque pontos positivos/negativos em um frame do plano (mais preciso que texto quando várias pessoas são parecidas)", "Marca puntos positivos/negativos en un fotograma del plano (más preciso que el texto si varias personas se parecen)", "在此镜头的一帧上标记正向/负向点（多人相似时比文本更准确）"],
  "Segment this shot (the points win over the text)": ["Segmentar este plano (os pontos têm prioridade sobre o texto)", "Segmentar este plano (los puntos tienen prioridad sobre el texto)", "分割此镜头（标记点优先于文本）"], "Segment this shot with the text and show a few frames": ["Segmentar este plano pelo texto e mostrar alguns frames", "Segmentar este plano con el texto y mostrar algunos fotogramas", "根据文本分割此镜头并预览若干帧"],
  "Colored Mark needs a region selection: enter a description, add points, or choose a mask video, then preview it.": ["A Marcação colorida precisa de uma região: digite uma descrição, adicione pontos ou escolha um vídeo de máscara e gere a prévia.", "La Marca de color necesita una región: escribe una descripción, añade puntos o elige un vídeo de máscara y genera la vista previa.", "彩色标记需要选定区域：输入描述、添加标记点或选择蒙版视频，然后预览。"],
  "nothing found: the shot runs without a mask": ["nada encontrado: o plano será executado sem máscara", "no se encontró nada: el plano se ejecutará sin máscara", "未找到目标：此镜头将不使用蒙版运行"],
  "The whole frame is regenerated. The mask, if any, only feeds {target} / the setting picture.": ["O frame inteiro é regenerado. Se houver máscara, ela só alimenta {target} ou a imagem de cenário.", "Se regenera todo el fotograma. Si hay máscara, solo se usa para {target} o la imagen del entorno.", "整帧都会重新生成。若有蒙版，仅用于 {target} 或场景参考图。"],
  "What the NEW subject is, in English, for SAM 3 on the result: BFS Shot Join also pastes its outline (when it is bigger than the old one). E.g. person, dog, cat, car. Empty = person.": ["Descreva em inglês o NOVO sujeito no resultado para o SAM 3: o BFS Shot Join também cola seu contorno quando ele é maior que o antigo. Ex.: person, dog, cat, car. Vazio = person.", "Describe en inglés el NUEVO sujeto del resultado para SAM 3: BFS Shot Join también pega su silueta cuando supera la anterior. Ej.: person, dog, cat, car. Vacío = person.", "用英文描述结果中的新主体，供 SAM 3 使用：若新轮廓更大，BFS Shot Join 也会粘贴其轮廓。例如 person、dog、cat、car。留空 = person。"],
  "Write {target} in the prompt: it becomes this description, so the model knows WHICH person to replace (useful with several people on screen).": ["Use {target} no prompt para inserir esta descrição e indicar ao modelo QUEM substituir (útil quando há várias pessoas na tela).", "Escribe {target} en el prompt para insertar esta descripción e indicar a quién debe sustituir el modelo (útil si hay varias personas).", "在提示词中使用 {target} 插入此描述，明确告诉模型要替换谁（画面中有多人时很有用）。"],
  "Uses this shot's selection (🎯 Points… on the person — click the BODY, not only the face, so the outfit is described — or the mask text): SAM 3 cuts the person out and the VLM describes them": ["Usa a seleção deste plano (🎯 Pontos… no corpo, não apenas no rosto, para descrever a roupa; ou o texto da máscara): o SAM 3 recorta a pessoa e o VLM a descreve.", "Usa la selección de este plano (🎯 Puntos… en el cuerpo, no solo en la cara, para describir la ropa; o el texto de máscara): SAM 3 recorta a la persona y el VLM la describe.", "使用此镜头的选区（🎯 在人物身体上标点，不要只点脸，这样才能描述服装；或使用蒙版文本）：SAM 3 抠出人物，VLM 生成描述。"],
  "Skip this shot": ["Ignorar este plano", "Omitir este plano", "跳过此镜头"], "unparsed answer:": ["resposta não interpretada:", "respuesta sin procesar:", "未解析的回答："],
  "previous shot as reference": ["plano anterior como referência", "plano anterior como referencia", "将上一镜头作为参考图"], "previous shot as first frame": ["plano anterior como primeiro frame", "plano anterior como primer fotograma", "将上一镜头作为首帧"], "previous shot's end as history (long take)": ["final do plano anterior como histórico (plano contínuo)", "final del plano anterior como historial (toma continua)", "将上一镜头的结尾作为历史（连续长镜头）"],
  "its first frame": ["seu primeiro frame", "su primer fotograma", "其首帧"], "its last frame": ["seu último frame", "su último fotograma", "其末帧"], "its middle frame": ["seu frame do meio", "su fotograma central", "其中间帧"],
  "Pick a video in the Video tab.": ["Escolha um vídeo na aba Vídeo.", "Elige un vídeo en la pestaña Vídeo.", "请在 Video 标签页中选择视频。"],
  "✓ copied to {…} shot(s)": ["✓ copiado para {…} plano(s)", "✓ copiado a {…} plano(s)", "✓ 已复制到 {…} 个镜头"], "show mask previews": ["mostrar prévias das máscaras", "mostrar vistas previas de las máscaras", "显示蒙版预览"],
  "max total seconds": ["duração total máxima", "duración total máxima", "总时长上限"],
  "Remove every reference picture from this plan (global, every shot) and forget the recent list?": ["Remover todas as imagens de referência deste plano (global e de todos os planos) e limpar a lista recente?", "¿Eliminar todas las imágenes de referencia de este plan (globales y de todos los planos) y borrar la lista reciente?", "移除此计划中的所有参考图（全局及各镜头）并清空最近列表？"],
  "Colored Mark": ["Marcação colorida", "Marca de color", "彩色标记"], "Default Colored Mark for this global reference": ["Cor padrão da Marcação colorida para esta referência global", "Color predeterminado de Marca de color para esta referencia global", "此全局参考图的默认彩色标记颜色"],
  "write {details} in any prompt: each shot gets the description of its own references": ["Use {details} em qualquer prompt para inserir a descrição das referências de cada plano", "Escribe {details} en cualquier prompt para insertar la descripción de las referencias de cada plano", "在提示词中使用 {details}，即可插入各镜头对应参考图的描述"],
  "e.g. the person on the left": ["ex.: a pessoa à esquerda", "p. ej., la persona de la izquierda", "例如：左侧的人物"], "Keeps movement and expression visible.": ["Mantém os movimentos e as expressões visíveis.", "Mantiene visibles el movimiento y la expresión.", "保留动作和表情的可见度。"],
  "What to segment, in English: a short phrase (noun + one or two traits), e.g. 'woman in pink top'; commas for several things. Enough on its own - no points needed: SAM 3 finds and tracks it through the shot (every match, up to Max objects). Press Enter to preview. Points, when set, take priority.": ["O que segmentar, em inglês: uma frase curta (substantivo e uma ou duas características), por exemplo, 'woman in pink top'; use vírgulas para vários objetos. Não precisa de pontos: o SAM 3 encontra e rastreia o objeto no plano (até Máx. objetos). Pressione Enter para pré-visualizar. Os pontos têm prioridade.", "Qué segmentar, en inglés: una frase breve (sustantivo y uno o dos rasgos), p. ej., 'woman in pink top'; usa comas para varios objetos. No hacen falta puntos: SAM 3 lo encuentra y sigue durante el plano (hasta Máx. objetos). Pulsa Enter para previsualizar. Los puntos tienen prioridad.", "用英文描述要分割的对象：简短短语（名词加一两个特征），例如“woman in pink top”；多个对象用逗号分隔。无需标记点，SAM 3 会在镜头中查找并跟踪（最多为最大对象数）。按 Enter 预览。设置了标记点时以标记点为准。"],
  "Remove this shot's mask (points, text, mask video) and mode - for shots where it picked the wrong thing": ["Remover a máscara (pontos, texto ou vídeo) e o modo deste plano — útil quando a seleção está errada", "Quitar la máscara (puntos, texto o vídeo) y el modo de este plano; útil si seleccionó el objeto equivocado", "移除此镜头的蒙版（点、文本或蒙版视频）及模式；适用于选错对象时"],
  "A black/white video (white = the subject), e.g. rotoscoped in another tool. It replaces SAM 3 for this shot and covers only this shot: its first frame is the shot's first frame.": ["Vídeo em preto e branco (branco = sujeito), por exemplo, rotoscopiado em outra ferramenta. Substitui o SAM 3 apenas neste plano; o primeiro frame do vídeo corresponde ao primeiro frame do plano.", "Vídeo en blanco y negro (blanco = sujeto), por ejemplo, rotoscopiado en otra herramienta. Sustituye SAM 3 solo en este plano; su primer fotograma corresponde al primer fotograma del plano.", "黑白蒙版视频（白色表示主体），例如在其他工具中制作的 rotoscoping。它仅在此镜头替代 SAM 3，视频首帧对应镜头首帧。"],
  "— none: SAM 3 (text / points) —": ["— nenhum: SAM 3 (texto / pontos) —", "— ninguno: SAM 3 (texto / puntos) —", "— 无：SAM 3（文本/标记点）—"], "— none: the global mask video (People / Filters / Masks) —": ["— nenhum: vídeo de máscara global (Pessoas / Filtros / Máscaras) —", "— ninguno: vídeo de máscara global (Personas / Filtros / Máscaras) —", "— 无：全局蒙版视频（人物/筛选/蒙版）—"],
  "upload a mask video for this shot": ["enviar um vídeo de máscara para este plano", "subir un vídeo de máscara para este plano", "为此镜头上传蒙版视频"], "back to SAM 3 / the global mask": ["voltar ao SAM 3 / máscara global", "volver a SAM 3 / máscara global", "切换回 SAM 3/全局蒙版"],
  "This shot uses its mask video: the text and points are ignored.": ["Este plano usa o vídeo de máscara; o texto e os pontos são ignorados.", "Este plano usa su vídeo de máscara; se ignoran el texto y los puntos.", "此镜头使用自己的蒙版视频，文本和标记点将被忽略。"],
  "No mask of its own: this shot uses the global mask video (People / Filters / Masks tab).": ["Sem máscara própria: este plano usa o vídeo de máscara global (aba Pessoas / Filtros / Máscaras).", "Sin máscara propia: este plano usa el vídeo de máscara global (pestaña Personas / Filtros / Máscaras).", "没有单独蒙版：此镜头使用全局蒙版视频（People / Filters / Masks 标签页）。"],
  "No mask of its own: this shot uses its part of the whole-video track (People / Filters / Masks tab).": ["Sem máscara própria: este plano usa sua parte do rastreamento do vídeo inteiro (aba Pessoas / Filtros / Máscaras).", "Sin máscara propia: este plano usa su parte del seguimiento de todo el vídeo (pestaña Personas / Filtros / Máscaras).", "没有单独蒙版：此镜头使用整段视频跟踪中对应的部分（People / Filters / Masks 标签页）。"],
  "Opacity of the generation mask: the value of the white inside the mask (1 = fully white, the default). H3 reads a grey mask value as 'regenerate this much': at 0.85 the masked area starts at 85% of the noise and keeps ~15% of the original there. 0.8-0.9 keeps pose, outline and lighting while still swapping; too low copies the original person.": ["Opacidade da máscara de geração: valor do branco dentro da máscara (1 = branco total, padrão). O H3 interpreta o cinza como intensidade de regeneração: em 0,85, a região começa com 85% de ruído e preserva cerca de 15% do original. Valores de 0,8–0,9 mantêm pose, contorno e iluminação; valores muito baixos copiam a pessoa original.", "Opacidad de la máscara de generación: valor del blanco dentro de la máscara (1 = blanco total, predeterminado). H3 interpreta el gris como intensidad de regeneración: con 0,85, la zona empieza con un 85 % de ruido y conserva cerca del 15 % del original. Valores de 0,8–0,9 mantienen pose, silueta e iluminación; valores demasiado bajos copian a la persona original.", "生成蒙版的不透明度：蒙版白色区域的数值（1 = 全白，为默认值）。H3 将灰度解释为重绘强度：0.85 表示区域从 85% 噪声开始，约保留 15% 原图。0.8–0.9 可保留姿势、轮廓和光线；过低会复制原人物。"],
  "keeps ~{…}% of the original inside the mask": ["preserva ~{…}% do original dentro da máscara", "conserva ~{…}% del original dentro de la máscara", "保留蒙版内约 {…}% 的原图"], "regenerates the masked area completely": ["regenera totalmente a região mascarada", "regenera por completo la zona enmascarada", "完全重绘蒙版区域"],
  "set BFS Shot H3 Conditioning → inpaint to ": ["configure BFS Shot H3 Conditioning → inpaint como ", "configura BFS Shot H3 Conditioning → inpaint como ", "将 BFS Shot H3 Conditioning → inpaint 设置为 "],
  "its outline on the result is pasted too, so a bigger silhouette is not cut": ["o contorno também é colado no resultado para não cortar uma silhueta maior", "también se pega su silueta en el resultado para no recortar una figura más grande", "结果中也会贴回其轮廓，避免裁掉更大的主体"],
  "who is replaced, e.g. the young woman in a pink crop top (empty = the person)": ["quem será substituído, por exemplo, a jovem de top rosa (vazio = a pessoa)", "quién se sustituye, p. ej., la joven con top rosa (vacío = la persona)", "要替换谁，例如穿粉色短上衣的年轻女性（留空 = 人物）"],
  "copy (the {shot} placeholder in the prompt gets it automatically at run time)": ["copiar (o marcador {shot} do prompt recebe isso durante a execução)", "copiar (el marcador {shot} del prompt lo inserta automáticamente al ejecutar)", "复制（运行时会自动填入提示词中的 {shot}）"],
  "SAM 3 mask (text or points), crop, {target}": ["Máscara SAM 3 (texto ou pontos), recorte, {target}", "Máscara SAM 3 (texto o puntos), recorte, {target}", "SAM 3 蒙版（文本或点）、裁剪、{target}"],
  "How much of the previous result's end goes before this shot. 5 frames is a short context; longer windows carry more motion and use more memory.": ["Quantos frames finais do resultado anterior são colocados antes deste plano. 5 frames dão pouco contexto; janelas maiores carregam mais movimento e usam mais memória.", "Cuántos fotogramas finales del resultado anterior se colocan antes de este plano. 5 fotogramas dan poco contexto; ventanas más largas conservan más movimiento y usan más memoria.", "将上一结果的多少末尾帧放在此镜头之前。5 帧是较短上下文；更长窗口保留更多动作，也占用更多内存。"],
  "Which frame of the previous shot's result": ["Qual frame do resultado do plano anterior", "Qué fotograma del resultado del plano anterior", "上一镜头结果中的哪一帧"],
  "the first shot uses only its references": ["o primeiro plano usa apenas suas referências", "el primer plano solo usa sus referencias", "第一个镜头仅使用自己的参考图"],
  "⚠ a camera cut starts this shot: history is skipped (it only carries a continuing take)": ["⚠ este plano começa após um corte de câmera: o histórico é ignorado (ele só funciona em uma tomada contínua)", "⚠ este plano empieza tras un corte de cámara: se omite el historial (solo sirve para una toma continua)", "⚠ 此镜头从镜头切换开始：跳过历史帧（历史帧只适用于连续长镜头）"],
  "the end of the PREVIOUS shot's result continues into this one (queue loop: shots run in order). Works with every mask mode, crop and the duet canvas.": ["o final do resultado do plano ANTERIOR continua neste plano (loop em fila: planos executados em ordem). Funciona com todos os modos de máscara, recorte e tela duet.", "el final del resultado del plano ANTERIOR continúa en este (bucle en cola: los planos se ejecutan en orden). Funciona con todos los modos de máscara, recorte y lienzo duet.", "上一镜头结果的末尾延续到此镜头（队列循环按顺序运行）。支持所有蒙版模式、裁剪和 duet 画布。"],
  "uses a frame of the PREVIOUS shot's result (queue loop, or auto loop with BFS Shot H3 Duet)": ["usa um frame do resultado do plano ANTERIOR (loop em fila ou loop automático com BFS Shot H3 Duet)", "usa un fotograma del resultado del plano ANTERIOR (bucle en cola o automático con BFS Shot H3 Duet)", "使用上一镜头结果中的一帧（队列循环，或搭配 BFS Shot H3 Duet 的自动循环）"],
  "the points, on each shot's frame at the same relative position (a subject that stays in place)": ["os pontos em cada plano, na mesma posição relativa do frame (para um sujeito que permanece no lugar)", "los puntos en cada plano, en la misma posición relativa del fotograma (para un sujeto que no se desplaza)", "在每个镜头相同相对位置的帧上放置这些点（适用于位置不变的主体）"],
  "the shot's mask video (it covers only its own shot: usually you want a different one per shot)": ["o vídeo de máscara do plano (cobre apenas este plano; normalmente cada plano precisa de um vídeo diferente)", "el vídeo de máscara del plano (solo cubre este plano; normalmente se usa uno distinto por plano)", "此镜头的蒙版视频（仅覆盖此镜头；通常每个镜头需要不同的视频）"],
  "the generation mode (Full frame / Frame + paste / Mask only / Crop / Crop + mask) and its mask strength (shots with a mask)": ["o modo de geração (Quadro inteiro / Quadro + colar / Somente máscara / Recorte / Recorte + máscara) e a força da máscara", "el modo de generación (Fotograma completo / Fotograma + pegar / Solo máscara / Recorte / Recorte + máscara) y la intensidad de máscara", "生成模式（整帧/整帧+贴回/仅蒙版/裁剪/裁剪+蒙版）及蒙版强度"],
  "the {target} description": ["a descrição de {target}", "la descripción de {target}", "{target} 的描述"], "continuity (not on the first shot)": ["continuidade (exceto no primeiro plano)", "continuidad (excepto en el primer plano)", "连续性（第一个镜头除外）"],
  "shots after this": ["planos depois deste", "planos posteriores a este", "此镜头之后的镜头"], "replaces the old “→ all” buttons": ["substitui os antigos botões “→ todos”", "sustituye los antiguos botones “→ todos”", "替代旧的“→ 全部”按钮"],
  "remove (use the global reference)": ["remover (usar a referência global)", "quitar (usar la referencia global)", "移除（使用全局参考图）"], "previous shot (←)": ["plano anterior (←)", "plano anterior (←)", "上一镜头（←）"],
  "Override the content filters for this shot": ["Substituir os filtros de conteúdo neste plano", "Anular los filtros de contenido para este plano", "覆盖此镜头的内容筛选设置"],
  "Colored Mark keeps two independent point sets; switch regions to edit each person. Both use the frame selected below.": ["Com Marcação colorida, cada pessoa tem seu próprio conjunto de pontos. Alterne as regiões para editar cada pessoa; ambas usam o frame selecionado abaixo.", "Marca de color mantiene un conjunto de puntos independiente por persona. Cambia de región para editar cada una; ambas usan el fotograma seleccionado abajo.", "彩色标记为每个人保留独立的点集。切换区域即可编辑；两组都使用下方选定的帧。"],
  "Describe second region": ["Descreva a segunda região", "Describe la segunda región", "描述第二个区域"],
  "🎯 Mark with points": ["🎯 Marcar com pontos", "🎯 Marcar con puntos", "🎯 用标记点选择"],
  "🎯 Points 2": ["🎯 Pontos 2", "🎯 Puntos 2", "🎯 标记点 2"],
  "Region 1 · color": ["Região 1 · cor", "Región 1 · color", "区域 1 · 颜色"],
  "Region 2 · color": ["Região 2 · cor", "Región 2 · color", "区域 2 · 颜色"],
  "click = keep (green) · right-click or shift+click = exclude (red) · click a point to remove it": ["clique = incluir (verde) · clique direito ou shift+clique = excluir (vermelho) · clique em um ponto para removê-lo", "clic = incluir (verde) · clic derecho o shift+clic = excluir (rojo) · clic en un punto para quitarlo", "单击 = 保留（绿色）· 右键或 shift+单击 = 排除（红色）· 单击标记点可删除"],
};
export function t(value) {
  const s = String(value ?? ""), lang = uiLanguage.value;
  if (lang === "en") return s;
  const trimmed = s.trim(), entry = TEXT[s] || TEXT[trimmed];
  if (entry) return `${s.slice(0, s.indexOf(trimmed))}${entry[{ pt: 0, es: 1, zh: 2 }[lang]]}${s.slice(s.indexOf(trimmed) + trimmed.length)}`;
  let m;
  if ((m = s.match(/^Person (\d+)$/))) return `${t("Person")} ${m[1]}`;
  if ((m = s.match(/^Region (\d+) · color$/))) return `${lang === "pt" ? "Região" : lang === "es" ? "Región" : "区域"} ${m[1]} · ${lang === "zh" ? "颜色" : lang === "pt" ? "cor" : "color"}`;
  if ((m = s.match(/^Region (\d+) \((\d+)\)$/))) return `${lang === "pt" ? "Região" : lang === "es" ? "Región" : "区域"} ${m[1]} (${m[2]})`;
  if ((m = s.match(/^🎯 Points 2 \((\d+)\)$/))) return `🎯 ${lang === "pt" ? "Pontos" : lang === "es" ? "Puntos" : "标记点"} 2 (${m[1]})`;
  if ((m = s.match(/^Shot #(\d+)( · points)?$/))) return `${t("Shot")} #${m[1]}${m[2] ? ` · ${lang === "pt" ? "pontos" : lang === "es" ? "puntos" : "点"}` : ""}`;
  if ((m = s.match(/^(\d+) people · (\d+) linked$/))) return lang === "pt" ? `${m[1]} pessoas · ${m[2]} vinculadas` : lang === "es" ? `${m[1]} personas · ${m[2]} vinculadas` : `${m[1]} 人物 · 已关联 ${m[2]} 人`;
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
  if ((m = s.match(/^first seen (.+) \(click to seek\)$/))) return lang === "pt" ? `visto primeiro em ${m[1]} (clique para ir até lá)` : lang === "es" ? `primera aparición ${m[1]} (haz clic para ir)` : `首次出现于 ${m[1]}（点击跳转）`;
  if ((m = s.match(/^(\d+) people · (\d+) included$/))) return lang === "pt" ? `${m[1]} pessoas · ${m[2]} incluídas` : lang === "es" ? `${m[1]} personas · ${m[2]} incluidas` : `${m[1]} 人物 · 已包含 ${m[2]} 人`;
  if ((m = s.match(/^(\d+) point\(s\) at frame (\d+)$/))) return lang === "pt" ? `${m[1]} ponto(s) no frame ${m[2]}` : lang === "es" ? `${m[1]} punto(s) en el fotograma ${m[2]}` : `第 ${m[2]} 帧有 ${m[1]} 个标记点`;
  if ((m = s.match(/^mask covers ([\d.]+)%(.+)$/))) return lang === "pt" ? `máscara cobre ${m[1]}%${m[2]}` : lang === "es" ? `la máscara cubre ${m[1]}%${m[2]}` : `蒙版覆盖 ${m[1]}%${m[2]}`;
  if ((m = s.match(/^last (\d+) frames \(([\d.]+) s\)$/))) return lang === "pt" ? `últimos ${m[1]} frames (${m[2]} s)` : lang === "es" ? `últimos ${m[1]} fotogramas (${m[2]} s)` : `最后 ${m[1]} 帧（${m[2]} 秒）`;
  if ((m = s.match(/^Mask opacity (.+)$/))) return lang === "pt" ? `Opacidade da máscara ${m[1]}` : lang === "es" ? `Opacidad de máscara ${m[1]}` : `蒙版不透明度 ${m[1]}`;
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
  let p = localizeProps(props);
  // Many controls commit on change (blur). Agent polling and progress events
  // must not replace their unsaved DOM draft while the user is still typing.
  if (p && typeof p === "object" && !Array.isArray(p) && Object.hasOwn(p, "value") &&
      (type === "textarea" || (type === "input" && [undefined, "text", "number", "search", "url", "email", "password"].includes(p.type)))) {
    const beforeUpdate = p.onVnodeBeforeUpdate;
    p = { ...p, onVnodeBeforeUpdate(vnode, previous) {
      if (Array.isArray(beforeUpdate)) beforeUpdate.forEach(fn => fn(vnode, previous));
      else beforeUpdate?.(vnode, previous);
      const el = previous.el;
      if (el && typeof document !== "undefined" && document.activeElement === el) {
        vnode.props.value = el.value;
      }
    } };
  }
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
  skip_fill: "original", audio_mode: "auto", ref_details: {}, picture_details: {}, auto_prompt: {}, auto_prompts: {}, mask_cfg: {}, vlm_cfg: {}, cast: {}, cast_split: false, cast_only: false,
  filters: { person: false, min_person_area: 0, max_persons: 0, face: false, skip_dark: false, dark_level: 0.06,
             skip_static: false, static_level: 0.004, min_frames: 0, samples: 6 },
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
export const MAX_REFS = 12;
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
