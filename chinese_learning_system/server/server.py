#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import os
import json
import uuid
import datetime
import logging
import traceback
from werkzeug.utils import secure_filename

# Import our utility modules
from utils.rag import initialize_models, initialize_rag_system
from utils.models import StudentProfile, LearningLog, Question, Test, LearningPath
from utils.storage import save_json, load_json, ensure_dir_exists

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Create required directories
ensure_dir_exists("data")
ensure_dir_exists("vectorstore")
ensure_dir_exists("student_profiles")
ensure_dir_exists("learning_logs")
ensure_dir_exists("tmp")

# Initialize RAG system
try:
    chat_model, embedding = initialize_models()
    retriever = initialize_rag_system(embedding)
except Exception as e:
    logger.error(f"Error initializing RAG system: {str(e)}")
    logger.error(traceback.format_exc())
    # We'll still start the server but with limited functionality
    chat_model, embedding, retriever = None, None, None

# Global variables
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Configure app
app.config['UPLOAD_FOLDER'] = os.path.abspath("data")
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve static files 
@app.route('/')
def serve_index():
    return send_from_directory('../', 'index.html')

@app.route('/css/<path:path>')
def serve_css(path):
    return send_from_directory('../css', path)

@app.route('/js/<path:path>')
def serve_js(path):
    return send_from_directory('../js', path)

# Serve uploaded files
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API Routes
@app.route('/api/student/profile', methods=['GET'])
def get_student_profile():
    try:
        # Check if there's an existing profile
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if not profiles:
            return jsonify({"error": "No profile found"}), 404
        
        # For simplicity, just return the first profile
        profile_path = os.path.join("student_profiles", profiles[0])
        profile_data = load_json(profile_path)
        
        return jsonify(profile_data)
    except Exception as e:
        logger.error(f"Error getting student profile: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/student/profile', methods=['POST'])
def save_student_profile():
    try:
        profile_data = request.json
        
        # Check if there's an existing profile
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if profiles:
            # Update existing profile
            profile_path = os.path.join("student_profiles", profiles[0])
            existing_data = load_json(profile_path)
            
            # Update fields
            existing_data["name"] = profile_data.get("name", existing_data.get("name", ""))
            existing_data["learning_style"] = profile_data.get("learning_style", existing_data.get("learning_style", ""))
            existing_data["interests"] = profile_data.get("interests", existing_data.get("interests", []))
            
            save_json(existing_data, profile_path)
        else:
            # Create new profile
            student_id = str(uuid.uuid4())[:8]
            
            profile = StudentProfile(
                id=student_id,
                name=profile_data.get("name", ""),
                learning_style=profile_data.get("learning_style", ""),
                current_knowledge_level="初學者",
                strengths=[],
                areas_for_improvement=[],
                interests=profile_data.get("interests", []),
                learning_history=[]
            )
            
            save_json(profile.model_dump(), os.path.join("student_profiles", f"{student_id}.json"))
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error saving student profile: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/survey/learning-style', methods=['GET'])
def generate_learning_style_survey():
    try:
        # Check if RAG system is initialized
        if chat_model is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        # Generate a learning style survey using the chat model
        prompt = """你是一位專精於學習風格評估的教育專家。
        請設計一份簡短但有效的學習風格測驗問卷，包含 5 個多選題。
        每個問題應有 3 個選項，用於判定學生是否主要是：
        1. 視覺型學習者
        2. 聽覺型學習者
        3. 動覺型學習者
        
        請將你的回應格式化為一個JSON對象，包含問題數組，每個問題有文本和選項。例如：
        {
          "questions": [
            {
              "text": "當你學習新知識時，你偏好：",
              "options": [
                {"text": "看圖表和視覺輔助", "value": "視覺型"},
                {"text": "聆聽講解和討論", "value": "聽覺型"},
                {"text": "動手實作和體驗", "value": "動覺型"}
              ]
            }
          ]
        }
        
        請確保每個問題都有針對三種不同學習風格的選項。"""

        messages = [{"role": "user", "content": prompt}]
        response = chat_model.invoke(messages)
        
        # Parse the response as JSON
        survey_data = response.content
        
        return jsonify(survey_data)
    except Exception as e:
        logger.error(f"Error generating learning style survey: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/survey/learning-style', methods=['POST'])
def submit_learning_style_survey():
    try:
        answers = request.json.get("answers", [])
        
        if not answers:
            return jsonify({"error": "No answers provided"}), 400
        
        # Count preferences
        style_counts = {"視覺型": 0, "聽覺型": 0, "動覺型": 0}
        
        for answer in answers:
            value = answer.get("value")
            if value in style_counts:
                style_counts[value] += 1
        
        # Determine dominant learning style
        learning_style = max(style_counts, key=style_counts.get)
        
        # Update student profile
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if profiles:
            profile_path = os.path.join("student_profiles", profiles[0])
            profile_data = load_json(profile_path)
            
            profile_data["learning_style"] = learning_style
            
            # Add survey activity to learning history
            profile_data["learning_history"].append({
                "activity_type": "學習風格問卷",
                "timestamp": datetime.datetime.now().isoformat(),
                "result": learning_style
            })
            
            save_json(profile_data, profile_path)
        
        return jsonify({"learning_style": learning_style})
    except Exception as e:
        logger.error(f"Error submitting learning style survey: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    try:
        # Get student profile
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if not profiles:
            return jsonify({
                "knowledge_level": "初學者",
                "learning_style": "未確定",
                "progress": 0,
                "recent_activities": [],
                "recommendations": []
            })
        
        profile_path = os.path.join("student_profiles", profiles[0])
        profile_data = load_json(profile_path)
        
        # Calculate progress
        progress = 0
        if profile_data.get("learning_history"):
            # Simple progress calculation
            completed_activities = len(profile_data["learning_history"])
            if completed_activities >= 10:
                progress = 100
            else:
                progress = (completed_activities / 10) * 100
        
        # Get recent activities
        recent_activities = []
        if profile_data.get("learning_history"):
            for activity in sorted(profile_data["learning_history"], 
                                  key=lambda x: x.get("timestamp", ""), reverse=True)[:5]:
                
                activity_type = activity.get("activity_type", "")
                timestamp = activity.get("timestamp", "")
                
                # Format date
                if timestamp:
                    try:
                        date = datetime.datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
                    except:
                        date = timestamp
                else:
                    date = "未知時間"
                
                description = f"{activity_type}"
                if activity_type == "前測" or activity_type == "後測":
                    score = activity.get("score", "")
                    if score:
                        description += f" (得分: {score})"
                elif activity_type == "學習風格問卷":
                    result = activity.get("result", "")
                    if result:
                        description += f" (結果: {result})"
                
                recent_activities.append({
                    "date": date,
                    "description": description
                })
        
        # Generate recommendations
        recommendations = []
        if not profile_data.get("learning_style"):
            recommendations.append("完成學習風格問卷以獲取個人化建議")
        elif not any(a.get("activity_type") == "前測" for a in profile_data.get("learning_history", [])):
            recommendations.append("參加前測以評估您的知識水平")
        elif profile_data.get("current_knowledge_level") == "初學者":
            recommendations.append("專注於學習基礎概念和核心詞彙")
        elif profile_data.get("current_knowledge_level") == "中級":
            recommendations.append("挑戰更複雜的內容，加強實際應用")
        else:
            recommendations.append("嘗試教導他人或創建自己的學習材料來強化知識")
        
        # Return dashboard data
        dashboard_data = {
            "knowledge_level": profile_data.get("current_knowledge_level", "初學者"),
            "learning_style": profile_data.get("learning_style", "未確定"),
            "progress": progress,
            "recent_activities": recent_activities,
            "recommendations": recommendations
        }
        
        return jsonify(dashboard_data)
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/upload', methods=['POST'])
def upload_documents():
    try:
        global retriever  # Move the global declaration to the top of the function

        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({"error": "No files selected"}), 400
        
        processed_files = []
        failed_files = []
        
        # Ensure upload directory exists
        ensure_dir_exists(app.config['UPLOAD_FOLDER'])
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the file
                file.save(filepath)
                logger.info(f"Saved file: {filepath}")
                
                try:
                    # Rebuild the vector store with the new document if RAG system is initialized
                    if retriever is not None and embedding is not None:
                        retriever = initialize_rag_system(embedding, rebuild=True)
                    processed_files.append(filename)
                except Exception as e:
                    failed_files.append(filename)
                    logger.error(f"Error processing file {filename}: {str(e)}")
            else:
                failed_files.append(file.filename)
        
        return jsonify({
            "success": True,
            "processed": processed_files,
            "failed": failed_files
        })
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/tests/availability', methods=['GET'])
def get_test_availability():
    try:
        # Check if pretest is available
        pretest_available = False
        posttest_available = False
        
        # Pretest is available if student profile exists and has a learning style
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if profiles:
            profile_path = os.path.join("student_profiles", profiles[0])
            profile_data = load_json(profile_path)
            
            if profile_data.get("learning_style"):
                pretest_available = True
            
            # Posttest is available if pretest has been completed
            if profile_data.get("learning_history") and \
               any(a.get("activity_type") == "前測" for a in profile_data["learning_history"]):
                posttest_available = True
        
        return jsonify({
            "pretest_available": pretest_available,
            "posttest_available": posttest_available
        })
    except Exception as e:
        logger.error(f"Error checking test availability: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/tests/pretest', methods=['GET'])
def generate_pretest():
    try:
        # Check if RAG system is initialized
        if chat_model is None or retriever is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        # Generate a pretest using the RAG system
        prompt = """你是一位專精於教學評估設計的專家。
        根據提供的內容，設計一份前測（Pre-Test），以評估學生在該主題上的現有知識水平。
        
        請設計包含不同難度等級的問題：簡單、中等和困難。
        對於每個問題，請提供：
        1. 問題文本
        2. 四個多選選項（A, B, C, D）
        3. 正確答案
        4. 為什麼正確的解釋
        5. 難度等級
        
        你必須遵循以下嚴格的 JSON 格式：
        {
          "title": "前測：[主題]",
          "description": "此測驗將評估你對[主題]的現有知識",
          "questions": [
            {
              "question": "問題文本？",
              "choices": ["A. 選項 A", "B. 選項 B", "C. 選項 C", "D. 選項 D"],
              "correct_answer": "A. 選項 A",
              "explanation": "為什麼 A 是正確答案的解釋",
              "difficulty": "簡單"
            }
          ]
        }

        請根據提供的內容生成總共 5 個問題，且包含不同難度等級的問題。
        """
        
        # Get some documents from the retriever
        documents = retriever.get_relevant_documents("前測")
        context = "\n\n".join([doc.page_content for doc in documents])
        
        messages = [{"role": "user", "content": f"{prompt}\n\n內容：\n{context}"}]
        response = chat_model.invoke(messages)
        
        # Parse the response as JSON
        test_data = json.loads(response.content)
        
        return jsonify(test_data)
    except Exception as e:
        logger.error(f"Error generating pretest: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/tests/posttest', methods=['GET'])
def generate_posttest():
    try:
        # Check if RAG system is initialized
        if chat_model is None or retriever is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        # Get student profile for knowledge level
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if not profiles:
            return jsonify({"error": "No student profile found"}), 404
        
        profile_path = os.path.join("student_profiles", profiles[0])
        profile_data = load_json(profile_path)
        
        knowledge_level = profile_data.get("current_knowledge_level", "初學者")
        
        # Generate a posttest using the RAG system
        prompt = f"""你是一位專業的教學評估設計師。
        根據提供的學習模組內容和學生的當前知識水平，設計一份後測，以評估學生的學習成果。
        
        難度應與學生的當前水平相匹配：
        - 初學者：較多簡單問題（70%），一些中等問題（30%）
        - 中級者：一些簡單問題（30%），主要是中等問題（50%），一些困難問題（20%）
        - 高級者：一些中等問題（40%），主要是困難問題（60%）
        
        設計的問題應測試學生對內容的理解、應用和分析能力。
        
        對於每個問題，請提供：
        1. 問題文本
        2. 四個多選選項（A, B, C, D）
        3. 正確答案
        4. 為什麼正確的解釋
        5. 難度等級
        
        你必須遵循以下嚴格的 JSON 格式：
        {{
          "title": "後測：[主題]",
          "description": "此測驗將評估你對[主題]的學習成果",
          "questions": [
            {{
              "question": "問題文本？",
              "choices": ["A. 選項 A", "B. 選項 B", "C. 選項 C", "D. 選項 D"],
              "correct_answer": "A. 選項 A",
              "explanation": "為什麼 A 是正確答案的解釋",
              "difficulty": "簡單"
            }}
          ]
        }}

        學生的當前知識水平: {knowledge_level}
        根據學生的水平生成總共 5 個問題，並適當分配難度。
        """
        
        # Get some documents from the retriever
        documents = retriever.get_relevant_documents("後測")
        context = "\n\n".join([doc.page_content for doc in documents])
        
        messages = [{"role": "user", "content": f"{prompt}\n\n內容：\n{context}"}]
        response = chat_model.invoke(messages)
        
        # Parse the response as JSON
        test_data = json.loads(response.content)
        
        return jsonify(test_data)
    except Exception as e:
        logger.error(f"Error generating posttest: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/tests/<test_type>/submit', methods=['POST'])
def submit_test(test_type):
    try:
        if test_type not in ["pretest", "posttest"]:
            return jsonify({"error": "Invalid test type"}), 400
        
        answers = request.json.get("answers", [])
        
        if not answers:
            return jsonify({"error": "No answers provided"}), 400
        
        # Calculate score
        correct_count = sum(1 for answer in answers if answer.get("choiceIndex") == answer.get("correctChoiceIndex", -1))
        total_questions = len(answers)
        percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Determine knowledge level based on score
        if percentage >= 80:
            knowledge_level = "高級"
        elif percentage >= 50:
            knowledge_level = "中級"
        else:
            knowledge_level = "初學者"
        
        # Update student profile
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if profiles:
            profile_path = os.path.join("student_profiles", profiles[0])
            profile_data = load_json(profile_path)
            
            # Update knowledge level if it's a pretest or if the posttest shows improvement
            if test_type == "pretest":
                profile_data["current_knowledge_level"] = knowledge_level
            elif test_type == "posttest":
                current_level = profile_data.get("current_knowledge_level", "初學者")
                
                # Only update if there's improvement
                if (current_level == "初學者" and knowledge_level in ["中級", "高級"]) or \
                   (current_level == "中級" and knowledge_level == "高級"):
                    profile_data["current_knowledge_level"] = knowledge_level
            
            # Add test activity to learning history
            profile_data["learning_history"].append({
                "activity_type": "前測" if test_type == "pretest" else "後測",
                "timestamp": datetime.datetime.now().isoformat(),
                "score": f"{correct_count}/{total_questions}",
                "percentage": percentage,
                "knowledge_level": knowledge_level
            })
            
            save_json(profile_data, profile_path)
        
        # Generate feedback based on score
        feedback = ""
        next_steps = []
        
        if percentage >= 80:
            feedback = "優秀！您展示了對主題的深入理解。"
            next_steps = [
                "考慮更高級的學習材料",
                "嘗試教導他人以鞏固知識",
                "應用所學解決實際問題"
            ]
        elif percentage >= 50:
            feedback = "做得不錯！您理解了大部分概念，但仍有改進空間。"
            next_steps = [
                "重新檢視困難問題的相關內容",
                "尋找實際應用的機會",
                "參與討論以加深理解"
            ]
        else:
            feedback = "這個主題對您來說有點挑戰性，但別氣餒！"
            next_steps = [
                "專注於基礎概念",
                "尋求更多示例和解釋",
                "採用適合您學習風格的學習策略"
            ]
        
        return jsonify({
            "score": correct_count,
            "total": total_questions,
            "percentage": percentage,
            "knowledge_level": knowledge_level,
            "feedback": feedback,
            "next_steps": next_steps,
            "answers": [
                {
                    "questionIndex": answer.get("questionIndex"),
                    "choiceIndex": answer.get("choiceIndex"),
                    "correctChoiceIndex": answer.get("correctChoiceIndex", answer.get("choiceIndex")),
                    "explanation": answer.get("explanation", "")
                }
                for answer in answers
            ]
        })
    except Exception as e:
        logger.error(f"Error submitting test: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/modules', methods=['GET'])
def get_learning_modules():
    try:
        # Check if RAG system is initialized
        if chat_model is None or retriever is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        # Get student profile to personalize modules
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if not profiles:
            return jsonify([])
        
        profile_path = os.path.join("student_profiles", profiles[0])
        profile_data = load_json(profile_path)
        
        # Generate learning modules based on available documents
        prompt = f"""你是一位專精於個人化學習路徑設計的教育課程設計專家。
        根據學生檔案和可用的學習材料，創建適合學生的學習模組列表。
        
        學生資料：
        - 姓名：{profile_data.get("name", "未知")}
        - 學習風格：{profile_data.get("learning_style", "未知")}
        - 當前知識水平：{profile_data.get("current_knowledge_level", "初學者")}
        - 興趣：{", ".join(profile_data.get("interests", [])) or "未知"}
        
        請生成3-5個學習模組，每個模組包含：
        1. 模組ID（唯一標識）
        2. 標題
        3. 簡短描述
        4. 適合的知識水平
        
        將你的回應格式化為JSON陣列，例如：
        [
          {{
            "id": "module-1",
            "title": "模組標題",
            "description": "模組描述",
            "level": "初學者/中級/高級"
          }}
        ]
        
        確保模組適合學生的知識水平和學習風格。
        """
        
        # Get documents to generate module topics
        documents = retriever.get_relevant_documents("模組")
        context = "\n\n".join([doc.page_content for doc in documents[:5]])  # Limit context
        
        messages = [{"role": "user", "content": f"{prompt}\n\n可用內容：\n{context}"}]
        response = chat_model.invoke(messages)
        
        # Parse the response as JSON
        modules_data = json.loads(response.content)
        
        return jsonify(modules_data)
    except Exception as e:
        logger.error(f"Error getting learning modules: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/modules/<module_id>/content', methods=['GET'])
def get_module_content(module_id):
    try:
        # Check if RAG system is initialized
        if chat_model is None or retriever is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        # Get student profile to personalize content
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if not profiles:
            return jsonify({"error": "No student profile found"}), 404
        
        profile_path = os.path.join("student_profiles", profiles[0])
        profile_data = load_json(profile_path)
        
        # Generate personalized content for the module
        prompt = f"""你是一位專業的教學內容創作者。
        根據提供的模組主題以及學生的學習風格和知識水平，創建引人入勝的教學內容。
        
        你的內容應該：
        1. 針對學生的學習風格（視覺型、聽覺型或動覺型）進行量身定制
        2. 符合學生的知識水平
        3. 包含關鍵概念的清晰解釋
        4. 使用範例和比喻來說明要點
        5. 包含符合知識水平的練習活動
        6. 結構清晰，包含明確的段落和標題
        7. 使用 markdown 格式提高可讀性
        
        模組ID: {module_id}
        學生學習風格: {profile_data.get("learning_style", "未知")}
        學生知識水平: {profile_data.get("current_knowledge_level", "初學者")}
        """
        
        # Get relevant documents based on module ID
        documents = retriever.get_relevant_documents(module_id)
        context = "\n\n".join([doc.page_content for doc in documents])
        
        messages = [{"role": "user", "content": f"{prompt}\n\n相關內容：\n{context}"}]
        response = chat_model.invoke(messages)
        
        return jsonify({
            "module_id": module_id,
            "content": response.content
        })
    except Exception as e:
        logger.error(f"Error getting module content: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/discussion/topics', methods=['GET'])
def get_discussion_topics():
    try:
        # Check if RAG system is initialized
        if chat_model is None or retriever is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        # Generate discussion topics based on available documents
        prompt = """你是一位專精於設計教育討論主題的專家。
        根據可用的學習材料，創建引人深思的討論主題列表。
        
        每個主題應該：
        1. 與學習材料相關
        2. 促進批判性思考
        3. 鼓勵學生應用所學知識
        4. 引發多角度思考
        
        將你的回應格式化為JSON陣列，例如：
        [
          {
            "id": "topic-1",
            "title": "討論主題標題",
            "description": "主題簡短描述"
          }
        ]
        
        請生成5-8個討論主題。
        """
        
        # Get documents to generate discussion topics
        documents = retriever.get_relevant_documents("討論")
        context = "\n\n".join([doc.page_content for doc in documents[:5]])  # Limit context
        
        messages = [{"role": "user", "content": f"{prompt}\n\n可用內容：\n{context}"}]
        response = chat_model.invoke(messages)
        
        # Parse the response as JSON
        topics_data = response.content
        
        return jsonify(topics_data)
    except Exception as e:
        logger.error(f"Error getting discussion topics: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/discussion/topics/<topic_id>', methods=['GET'])
def get_discussion_topic(topic_id):
    try:
        # Check if RAG system is initialized
        if chat_model is None or retriever is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        # Generate topic information
        prompt = f"""你是一位專精於設計教育討論的專家。
        請提供有關以下討論主題的詳細資訊：
        
        主題ID: {topic_id}
        
        你的回應應包含：
        1. 主題標題
        2. 詳細描述
        3. 學習目標
        4. 初始訊息（AI學習夥伴的第一條訊息）
        
        將你的回應格式化為JSON對象，例如：
        {{
          "id": "{topic_id}",
          "title": "討論主題標題",
          "description": "詳細描述",
          "objectives": ["目標1", "目標2", "目標3"],
          "initial_message": "AI學習夥伴的第一條訊息，引導學生開始討論"
        }}
        """
        
        # Get relevant documents based on topic ID
        documents = retriever.get_relevant_documents(topic_id)
        context = "\n\n".join([doc.page_content for doc in documents])
        
        messages = [{"role": "user", "content": f"{prompt}\n\n相關內容：\n{context}"}]
        response = chat_model.invoke(messages)
        
        # Parse the response as JSON
        topic_data = json.loads(response.content)
        
        return jsonify(topic_data)
    except Exception as e:
        logger.error(f"Error getting discussion topic: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/discussion/topics/<topic_id>/message', methods=['POST'])
def send_chat_message(topic_id):
    try:
        # Check if RAG system is initialized
        if chat_model is None or retriever is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        message = request.json.get("message", "")
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        # Get student profile
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if not profiles:
            return jsonify({"error": "No student profile found"}), 404
        
        profile_path = os.path.join("student_profiles", profiles[0])
        profile_data = load_json(profile_path)
        
        # Generate AI response using the discussion AI
        prompt = f"""你是「學習夥伴」，一個親切且有啟發性的 AI 同學，與學生進行有建設性的討論。
        你的角色是：
        1. 模擬一位也在學習該主題但有一定見解的同學
        2. 提出促進深度判斷性思考的開放問題
        3. 提供啟發性的指導，而不是直接給出答案
        4. 以對話的形式表達思考，就像是學生之間的交流
        5. 使用蘇格拉底式提問法幫助學生發現答案
        6. 鼓勵並保持良好的氛圍
        
        根據提供的相關內容回應，但不要只是簡單地重述資訊。
        而是以自然的方式進行交流，就像一起學習一樣。
        
        討論主題: {topic_id}
        學生學習風格: {profile_data.get("learning_style", "未知")}
        學生知識水平: {profile_data.get("current_knowledge_level", "初學者")}
        學生訊息: {message}
        """
        
        # Get relevant documents based on topic ID and message
        combined_query = f"{topic_id} {message}"
        documents = retriever.get_relevant_documents(combined_query)
        context = "\n\n".join([doc.page_content for doc in documents])
        
        messages = [{"role": "user", "content": f"{prompt}\n\n相關內容：\n{context}"}]
        response = chat_model.invoke(messages)
        
        return jsonify({
            "message": response.content
        })
    except Exception as e:
        logger.error(f"Error sending chat message: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_learning_logs():
    try:
        # Get all learning logs
        logs = []
        log_files = [f for f in os.listdir("learning_logs") if f.endswith('.json')]
        
        for log_file in log_files:
            log_path = os.path.join("learning_logs", log_file)
            log_data = load_json(log_path)
            logs.append(log_data)
        
        # Sort logs by timestamp
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Error getting learning logs: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/<log_id>', methods=['GET'])
def get_learning_log(log_id):
    try:
        log_path = os.path.join("learning_logs", f"{log_id}.json")
        
        if not os.path.exists(log_path):
            return jsonify({"error": "Log not found"}), 404
        
        log_data = load_json(log_path)
        
        return jsonify(log_data)
    except Exception as e:
        logger.error(f"Error getting learning log: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs', methods=['POST'])
def save_learning_log():
    try:
        log_data = request.json
        
        # Check required fields
        if not log_data.get("topic") or not log_data.get("content"):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get student profile
        profiles = [f for f in os.listdir("student_profiles") if f.endswith('.json')]
        
        if not profiles:
            return jsonify({"error": "No student profile found"}), 404
        
        profile_path = os.path.join("student_profiles", profiles[0])
        profile_data = load_json(profile_path)
        
        # Create or update log
        if log_data.get("id"):
            # Update existing log
            log_id = log_data["id"]
            log_path = os.path.join("learning_logs", f"{log_id}.json")
            
            if not os.path.exists(log_path):
                return jsonify({"error": "Log not found"}), 404
            
            existing_log = load_json(log_path)
            
            # Update fields
            existing_log["topic"] = log_data["topic"]
            existing_log["content"] = log_data["content"]
            existing_log["reflections"] = log_data.get("reflections", [])
            
            # Save updated log
            save_json(existing_log, log_path)
            
            log = existing_log
        else:
            # Create new log
            log_id = str(uuid.uuid4())[:8]
            
            log = {
                "id": log_id,
                "student_id": profile_data["id"],
                "timestamp": datetime.datetime.now().isoformat(),
                "topic": log_data["topic"],
                "content": log_data["content"],
                "reflections": log_data.get("reflections", []),
                "questions": [],
                "next_steps": []
            }
            
            # Save new log
            save_json(log, os.path.join("learning_logs", f"{log_id}.json"))
            
            # Add log activity to learning history
            profile_data["learning_history"].append({
                "activity_type": "學習日誌",
                "timestamp": log["timestamp"],
                "topic": log["topic"]
            })
            
            save_json(profile_data, profile_path)
        
        return jsonify(log)
    except Exception as e:
        logger.error(f"Error saving learning log: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/<log_id>/analyze', methods=['GET'])
def analyze_learning_log(log_id):
    try:
        # Check if RAG system is initialized
        if chat_model is None:
            return jsonify({"error": "RAG system not initialized"}), 500
            
        log_path = os.path.join("learning_logs", f"{log_id}.json")
        
        if not os.path.exists(log_path):
            return jsonify({"error": "Log not found"}), 404
        
        log_data = load_json(log_path)
        
        # Generate log analysis
        prompt = f"""你是一位專業的教學分析師，精於分析學生的學習日誌。
        根據學生的學習日誌，評估：
        
        1. 對關鍵概念的理解程度
        2. 優點和自信的領域
        3. 困惑或誤解的領域
        4. 對情感反應的感知
        5. 學習風格的指示
        
        將你的回應格式化為以下嚴格的 JSON 結構:
        {{
          "understanding_level": "高/中/低",
          "strengths": ["優點 1", "優點 2"],
          "areas_for_improvement": ["改進領域 1", "改進領域 2"],
          "emotional_response": "對情感反應的描述",
          "learning_style_indicators": ["指示 1", "指示 2"],
          "recommended_next_steps": ["建議步驟 1", "建議步驟 2"],
          "suggested_resources": ["資源 1", "資源 2"]
        }}
        """
        
        messages = [{"role": "user", "content": f"{prompt}\n\n學生: {log_data.get('student_id', 'unknown')}\n主題: {log_data['topic']}\n學習日誌內容:\n{log_data['content']}"}]
        response = chat_model.invoke(messages)
        
        # Parse the response as JSON
        analysis_data = json.loads(response.content)
        
        # Update the log with analysis results
        log_data["analysis"] = analysis_data
        save_json(log_data, log_path)
        
        return jsonify(analysis_data)
    except Exception as e:
        logger.error(f"Error analyzing learning log: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # Check if directories exist
        dirs_exist = all(os.path.exists(d) for d in ["data", "vectorstore", "student_profiles", "learning_logs"])
        
        # Check if RAG system is initialized
        rag_initialized = chat_model is not None and embedding is not None and retriever is not None
        
        return jsonify({
            "status": "healthy",
            "dirs_exist": dirs_exist,
            "rag_initialized": rag_initialized,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)