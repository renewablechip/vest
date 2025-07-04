import os
from flask import Flask, request, render_template, abort, Response, jsonify
import requests
import json
# sys.path.append(r'C:/yolo/yolov12-main/')
from ultralytics import YOLO
from PIL import Image as PILImage
import numpy as np
from datetime import datetime
# 在现有导入后添加以下导入
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.units import cm
from pathlib import Path
from io import BytesIO
import uuid

import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
app = Flask(__name__)

# 关闭浏览器缓存
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# --- 新增: 模型文件夹路径 ---
MODEL_FOLDER = 'model'
os.makedirs(MODEL_FOLDER, exist_ok=True) # 确保模型文件夹存在

# 设置上传与检测目录
UPLOAD_FOLDER = 'static/uploads'
DETECT_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)
# 在现有目录设置后添加
REPORT_DIR = 'static/reports'
FONT_PATH = 'fonts/msyh.ttc'  # 确保你有这个字体文件
os.makedirs(REPORT_DIR, exist_ok=True)
app.config['REPORT_DIR'] = REPORT_DIR

# 注册中文字体
try:
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", FONT_PATH))
except Exception as e:
    print(f"[字体注册失败]：{e}")

# 设置日志目录
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志格式 - 只记录检测相关的日志
# 创建专门的检测日志记录器
detection_logger = logging.getLogger('detection')
detection_logger.setLevel(logging.INFO)

# 创建日志处理器
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'detection.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
handler.setLevel(logging.INFO)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 添加处理器到检测日志记录器
detection_logger.addHandler(handler)

# 防止日志传播到根日志记录器
detection_logger.propagate = False

# 禁用Flask的默认日志记录
# logging.getLogger('werkzeug').setLevel(logging.ERROR)
# app.logger.setLevel(logging.ERROR)

# 使用检测日志记录器
logger = detection_logger

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER
# 在app配置部分添加DeepSeek API配置
DEEPSEEK_API_KEY = "sk-XXX"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


# 添加日志记录函数
def log_detection(model_name, image_filename, detections, detection_time, ip_address):
    """记录检测信息"""
    # 统计检测到的对象数量
    detected_count = len([d for d in detections if d != "No objects detected."])

    # 格式化检测结果为简洁字符串
    if detections == ["No objects detected."] or not detections:
        detection_summary = "未检测到对象"
    else:
        detection_summary = f"检测到{detected_count}个对象: " + ", ".join([d.split(':')[0] for d in detections])

    # 记录简洁的日志信息
    log_message = f"检测记录 | 时间:{detection_time} | 模型:{model_name} | 图片:{image_filename} | 结果:{detection_summary} | IP:{ip_address}"

    logger.info(log_message)


def generate_pdf(report_data: dict, original_image_path: str, annotated_image_path: str, output_pdf_path: str):
    """生成PDF报告"""
    # 注册中文字体
    font_path = FONT_PATH
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('SimHei', font_path))
            font_name = 'SimHei'
        except Exception as e:
            print(f"字体注册失败，使用默认字体: {e}")
            font_name = 'Helvetica'
    else:
        print("字体文件不存在")
        font_name = 'Helvetica'

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=cm, leftMargin=cm,
                            topMargin=cm, bottomMargin=cm)
    styles = getSampleStyleSheet()

    # 自定义样式
    if 'ReportTitle' not in styles.byName:
        styles.add(ParagraphStyle(name='ReportTitle', fontName=font_name, fontSize=24,
                                  alignment=TA_CENTER, textColor=colors.darkblue))
    if 'NormalCN' not in styles.byName:
        styles.add(ParagraphStyle(name='NormalCN', fontName=font_name, fontSize=12, leading=18))

    story = []

    # 添加标题段落
    story.append(Paragraph("AI检测报告", styles['ReportTitle']))
    story.append(Spacer(1, 0.5 * cm))

    # 基础信息
    story.append(Paragraph(f"<b>报告编号:</b> {report_data.get('report_id', '')}", styles['NormalCN']))
    story.append(Paragraph(f"<b>检测时间:</b> {report_data.get('detection_time', '')}", styles['NormalCN']))
    story.append(Paragraph(f"<b>使用模型:</b> {report_data.get('model_name', '')}", styles['NormalCN']))
    story.append(Spacer(1, 0.5 * cm))

    # 图像展示
    if os.path.exists(original_image_path) and os.path.exists(annotated_image_path):
        try:
            img1 = ReportLabImage(original_image_path, width=240, height=180)
            img2 = ReportLabImage(annotated_image_path, width=240, height=180)
            story.append(Paragraph("检测图像对比：", styles['NormalCN']))
            story.append(Table([["原始图像", "检测结果"]], colWidths=[240, 240]))
            story.append(Table([[img1, img2]], colWidths=[240, 240]))
            story.append(Spacer(1, 0.5 * cm))
        except Exception as e:
            print(f"图像处理错误: {e}")
            story.append(Paragraph("无法加载图像文件。", styles['NormalCN']))
    else:
        story.append(Paragraph("无法加载图像文件，请检查图片路径是否正确。", styles['NormalCN']))

    # 检测结果表格
    story.append(Paragraph("检测结果：", styles['NormalCN']))
    detections = report_data.get('detections', [])
    if detections and detections != ["No objects detected."]:
        data = [['编号', '类别', '置信度']]
        for i, d in enumerate(detections):
            parts = d.split(': ')
            class_name = parts[0] if len(parts) > 0 else d
            confidence = parts[1] if len(parts) > 1 else 'N/A'
            data.append([str(i + 1), class_name, confidence])

        table = Table(data, hAlign='CENTER')
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DDEEFF')),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("未检测到任何对象。", styles['NormalCN']))

    story.append(Spacer(1, 0.5 * cm))

    # AI建议部分（如果有的话）
    ai_advice = report_data.get('ai_advice', '')
    if ai_advice:
        story.append(Paragraph("AI医学建议：", styles['NormalCN']))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(ai_advice, styles['NormalCN']))
        story.append(Spacer(1, 0.3 * cm))

        disclaimer = "<b>重要提醒：以上建议由AI生成，仅供参考，不能替代专业医生的诊断和治疗建议。如有健康问题，请及时咨询专业医生。</b>"
        story.append(Paragraph(disclaimer, styles['NormalCN']))

    # 生成 PDF
    doc.build(story)

    # 写入文件
    with open(output_pdf_path, 'wb') as f:
        f.write(buffer.getvalue())


def get_medical_advice(detections):
    """
    以流式方式调用DeepSeek API并生成医学建议。
    这是一个生成器函数。
    """
    if not detections or detections == ["No objects detected."]:
        yield "data: 未检测到相关对象，无法提供建议。\n\n"
        return

    detection_text = ", ".join(detections)
    prompt = f"""
    基于以下医学影像检测结果，请提供专业的医学建议和分析，请直接回答，不要用markdown格式，也不要说“好的，这是一个基于...”这样的开场白：
    检测结果：{detection_text}
    请注意：这些建议仅供参考，不能替代专业医生的诊断，建议咨询专业医生。
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"  # 关键：告诉API我们想要流式响应
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的医学AI助手，能够基于医学影像检测结果提供专业的医学建议和分析。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
        "stream": True  # 关键：启用流式传输
    }

    try:
        # 使用 stream=True 来发起流式请求
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=90, stream=True)
        response.raise_for_status()

        # 逐块处理返回的数据
        for chunk in response.iter_lines():
            if chunk:
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith('data: '):
                    # 去掉 'data: ' 前缀
                    json_str = chunk_str[6:]
                    if json_str.strip() == '[DONE]':
                        # 流结束
                        break
                    try:
                        # 解析JSON数据
                        chunk_data = json.loads(json_str)
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            delta = chunk_data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                # 关键：将每个 token 包装成 SSE 格式并 yield
                                # SSE格式要求是 "data: <your_data>\n\n"
                                yield f"data: {json.dumps({'content': content})}\n\n"
                    except json.JSONDecodeError:
                        print(f"无法解析的JSON块: {json_str}")
                        continue
        # 发送一个完成信号（可选，但推荐）
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    except requests.exceptions.RequestException as e:
        print(f"DeepSeek API调用失败: {e}")
        error_msg = json.dumps({'error': 'AI建议服务暂时不可用，请稍后再试。'})
        yield f"data: {error_msg}\n\n"
    except Exception as e:
        print(f"处理AI建议时出错: {e}")
        error_msg = json.dumps({'error': '处理AI建议时出现错误。'})
        yield f"data: {error_msg}\n\n"

# 获取可用模型的辅助函数
def get_available_models():
    """扫描模型文件夹并返回所有.pt文件"""
    return [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')]

@app.route('/', methods=['GET', 'POST'])
def upload_detect():
    available_models = get_available_models()

    if request.method == 'POST':
        # --- 新增: 获取用户选择的模型 ---
        selected_model_name = request.form.get("model")
        if not selected_model_name or selected_model_name not in available_models:
            # 如果没有选择模型或选择的模型无效，则返回错误
            abort(400, "Invalid model selected.")

        # 获取客户端上传的图片
        image_file = request.files["image"]
        if image_file:
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + image_file.filename
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            detect_path = os.path.join(app.config["DETECT_FOLDER"], filename)

            image_file.save(upload_path)

            # --- 修改: 动态加载选择的模型 ---
            model_path = os.path.join(MODEL_FOLDER, selected_model_name)
            model = YOLO(model_path)

            # 使用YOLOv12进行目标检测
            results = model(upload_path)

            # 绘制检测结果图像并保存
            result_img_array = results[0].plot()
            result_pil = PILImage.fromarray(result_img_array)
            result_pil.save(detect_path)

            # 提取检测框信息（标签 + 置信度）
            detections = []
            boxes = results[0].boxes
            if boxes is not None and boxes.cls.numel() > 0:
                for cls_id, conf in zip(boxes.cls, boxes.conf):
                    class_name = model.names[int(cls_id)]
                    confidence = round(float(conf) * 100, 2)
                    detections.append(f"{class_name}: {confidence}%")
            else:
                detections.append("No objects detected.")

            # 【新增】记录检测日志
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            client_ip = request.remote_addr or "Unknown"
            log_detection(selected_model_name, filename, detections, detection_time, client_ip)

            return render_template(
                'index.html',
                prediction="Detection Complete",
                detections=detections,
                image_path=f"detections/{filename}",
                models = available_models,
                selected_model = selected_model_name  # 传递当前选择的模型，以便在页面上保持选中状态
            )

        # 对于GET请求，只传递模型列表
    return render_template('index.html', prediction=None, models=available_models)


@app.route('/get-ai-advice', methods=['POST'])
def get_ai_advice_route():
    """接收检测结果并以流式方式返回AI建议"""
    data = request.get_json()
    if not data or 'detections' not in data:
        abort(400, "请求中缺少检测数据。")

    detections = data['detections']

    # 关键：返回一个流式响应
    return Response(get_medical_advice(detections), mimetype='text/event-stream')


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_route():
    """生成PDF报告"""
    data = request.get_json()
    if not data:
        abort(400, "请求数据缺失。")

    # 获取必要的数据
    detections = data.get('detections', [])
    ai_advice = data.get('ai_advice', '')
    model_name = data.get('model_name', '')
    image_filename = data.get('image_filename', '')

    if not image_filename:
        abort(400, "缺少图像文件名。")

    # 生成唯一的PDF文件名
    pdf_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}.pdf"
    pdf_path = os.path.join(app.config['REPORT_DIR'], pdf_filename)

    # 图像路径
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    annotated_image_path = os.path.join(app.config['DETECT_FOLDER'], image_filename)

    # 构建报告数据
    report_data = {
        "report_id": str(uuid.uuid4())[:8].upper(),
        "detection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "detections": detections,
        "ai_advice": ai_advice
    }

    try:
        generate_pdf(report_data, original_image_path, annotated_image_path, pdf_path)
        return jsonify({
            'success': True,
            'pdf_url': f"/static/reports/{pdf_filename}",
            'filename': f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        })
    except Exception as e:
        print(f"PDF生成错误: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# 添加查看日志的路由
@app.route('/logs')
def view_logs():
    """查看检测日志"""
    try:
        log_file = os.path.join(LOG_DIR, 'detection.log')
        if not os.path.exists(log_file):
            return "日志文件不存在"

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.readlines()

        # 只显示最近的100条日志
        recent_logs = logs[-100:] if len(logs) > 100 else logs

        return render_template('logs.html', logs=recent_logs)
    except Exception as e:
        return f"读取日志失败: {str(e)}"


# 添加清理日志的路由（可选）
@app.route('/clear-logs', methods=['POST'])
def clear_logs():
    """清理日志文件"""
    try:
        log_file = os.path.join(LOG_DIR, 'detection.log')
        if os.path.exists(log_file):
            open(log_file, 'w').close()  # 清空文件
            logger.info("日志文件已清理")
            return jsonify({'success': True, 'message': '日志已清理'})
        else:
            return jsonify({'success': False, 'message': '日志文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'清理失败: {str(e)}'})

@app.route('/team')
def team_info():
    """显示小组成员信息"""
    return render_template('team.html')


if __name__ == '__main__':
    app.run(debug=True)
