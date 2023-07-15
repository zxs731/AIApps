from flask import Flask, request, render_template, jsonify
import MyAI as ai

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('mindmap.html')

@app.route('/api/mindmap')
def mindmap():
    # 返回思维导图的数据源
    data = {
        "meta": {
            "name": "个人知识管理",
            "author": "Flask",
            "version": "1.0"
        },
        "format": "node_array",
        "data": [
            {"id": "root", "isroot": "true", "topic": "个人知识管理"},
            '''
            {"id": "1", "parentid": "root", "topic": "节点1"},
            {"id": "2", "parentid": "root", "topic": "节点2"},
            {"id": "3", "parentid": "2", "topic": "节点3"},
            {"id": "4", "parentid": "2", "topic": "节点4"},
            {"id": "5", "parentid": "1", "topic": "节点5"},
            {"id": "6", "parentid": "1", "topic": "节点6"}
            '''
        ]
    }
    return jsonify(data)

@app.route('/api/generatekeywords', methods=['POST'])
def generatekeywords():
    data = request.get_json()
    topic = data['topic']
    keywords = ai.RelatedKeywords(text=topic)
    print(keywords)
    data=[]
    for t in keywords:
        data.append({'topic':t})
    
    return jsonify(data)

@app.route('/api/generateSearch', methods=['POST'])
def generateSearch():
    data = request.get_json()
    topic = data['topic']
    keywords = ai.GenerateHotItems(topic)
    print(keywords)
    data=[]
    for t in eval(keywords):
        data.append({'topic':t})
    
    return jsonify(data)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    topic = data['topic']
    keywords = ai.GetAnswer(topic=topic)
    print(keywords)
    data=[]
    for t in keywords:
        data.append({'topic':t})
    
    return jsonify(data)

@app.route('/api/questions', methods=['POST'])
def questions():
    data = request.get_json()
    topic = data['topic']
    keywords = ai.GenerateQuestions(topic=topic)
    print(keywords)
    data=[]
    for t in keywords:
        data.append({'topic':t})
    
    return jsonify(data)

@app.route('/api/article', methods=['POST'])
def article():
    data = request.get_json()
    topic = data['topic']
    keywords = ai.GenerateArticle(topic=topic)
    print(keywords)
    data=[]
    data.append({'topic':keywords})
    
    return jsonify(data)


@app.route('/api/hot')
def hot():
    keywords=ai.GenerateHot()
    data=[]
    for t in eval(keywords):
        data.append({'topic':t})
    
    return jsonify(data)

if __name__ == '__main__':
    app.run()