from flask import Flask, request, send_from_directory, render_template_string, jsonify, redirect, url_for
import numpy as np
import os
from compute_table import compute_t
import sqlite3
import pandas as pd
import numpy as np
import sqlite3
import pandas as pd
import numpy as np

app = Flask(__name__)

# ---- Placeholder model function -------------------------------------------------
def compute(x: np.ndarray) -> np.ndarray:
    return compute_t(x)


def login(user_id: str, password: str) -> bool:
    """简单的账号密码校验."""
    if user_id == "001" and password == "123":
        return True
    return False


def get_info_from_pid(pid="001"):
    """
    从 main_data 表中读取指定编号（pid）的信息，并返回一个 np.array。
    """
    # 连接数据库
    db_path = os.path.join(os.path.dirname(__file__), "ykdb.db")
    conn = sqlite3.connect(db_path)

    # 定义 SQL 查询语句
    query = f'''
    SELECT
      "眼别（1=OD 2=OS）",
      "性别（1=男 2=女）",
      "是否成人（0<15 Y）",
      "年龄",
      "检查年龄",
      "AL发育分组",
      "术后是否近视（0正视，近视=-1，远 视=1）",
      "是否弱视（是=1）",
      "Heart(1=正常型，2=反流型，3=器质型)",
      "Gene（11-FBN1半胱氨酸取代取代，12-FBN1生成半胱氨酸，13-FBN1其他，2-ADAMTSL4，3-ADAMTS17，4-ASPH，5-CBS，6CPAMD8，7SUOX）",
      "IOL度数",
      "手术方式 inbag=1 outofbag=2",
      "Pre-LogMar(术前眼部参数）",
      "pre-IOLMaster-AL(mm)",
      "pre-IOLMaser-K1(D)",
      "pre-IOLMaster-K1轴位",
      "pre-IOLMaster-K2(D)",
      "pre-IOLMaster-K2轴位",
      "pre-ACD",
      "pre-WTW",
      "Z-Pre WTW"
    FROM main_data
    WHERE 编号 = ?
    '''

    # 读取查询结果
    df = pd.read_sql_query(query, conn, params=(pid,))

    # 关闭连接
    conn.close()

    # 如果没有找到对应编号，返回空数组
    if df.empty:
        print(f"未找到编号 {pid} 的记录")
        return np.array([])

    # 将 DataFrame 转为 numpy 数组
    result = df.to_numpy()[0]

    return result


FIELDS = [
    ("eye", "眼别 (1=OD, 2=OS)"),
    ("sex", "性别 (1=男, 2=女)"),
    ("is_adult", "是否成人 (成人=1, <15Y=0)"),
    ("age", "年龄"),
    ("exam_age", "检查年龄"),
    ("al_group", "AL发育分组"),
    ("postop_myopia", "术后是否近视 (正视=0, 近视=-1, 远视=1)"),
    ("amblyopia", "是否弱视 (是=1, 否=0)"),
    ("heart", "Heart (1=正常型, 2=反流型, 3=器质型)"),
    ("gene", "Gene (11=FBN1半胱氨酸取代, 12=FBN1生成半胱氨酸, 13=FBN1其他, 2=ADAMTSL4, 3=ADAMTS17, 4=ASPH, 5=CBS, 6=CPAMD8, 7=SUOX)"),
    ("iol_power", "IOL度数"),
    ("surgery_type", "手术方式 (inbag=1, outofbag=2)"),
    ("pre_logmar", "Pre-LogMar(术前眼部参数)"),
    ("pre_al", "pre-IOLMaster-AL(mm)"),
    ("pre_k1", "pre-IOLMaster-K1(D)"),
    ("pre_k1_axis", "pre-IOLMaster-K1轴位"),
    ("pre_k2", "pre-IOLMaster-K2(D)"),
    ("pre_k2_axis", "pre-IOLMaster-K2轴位"),
    ("pre_acd", "pre-ACD"),
    ("pre_wtw", "pre-WTW"),
    ("z_pre_wtw", "Z-Pre WTW"),
]

@app.route('/html/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/html', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    values = {}
    result = None

    if request.method == 'POST':
        vector = []
        for key, _ in FIELDS:
            raw = request.form.get(key, '').strip()
            values[key] = raw
            try:
                vector.append(float(raw))
            except ValueError:
                vector.append(0.0)
        x = np.array(vector, dtype=float)
        y = compute(x)
        result = y.tolist()

    with open(os.path.join('static/html', 'template.html'), encoding='utf-8') as f:
        html_template = f.read()

    return render_template_string(html_template, fields=FIELDS, values=values, result=result)


@app.route('/login', methods=['GET', 'POST'])
def login_view():
    error = None

    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip()
        password = request.form.get('password', '')

        if login(user_id, password):
            return redirect(url_for('index'))

        error = "账号或密码错误，请重试。"

    with open(os.path.join('static/html', 'login.html'), encoding='utf-8') as f:
        html_template = f.read()

    return render_template_string(html_template, error=error)

@app.route('/api/prefill', methods=['GET'])
def api_prefill():
    pid = request.args.get('pid', '001').strip()
    try:
        arr = get_info_from_pid(pid)
        # 兜底保证长度与字段一致
        if arr.shape[0] < len(FIELDS):
            pad = np.zeros(len(FIELDS) - arr.shape[0], dtype=float)
            arr = np.concatenate([arr, pad])
        elif arr.shape[0] > len(FIELDS):
            arr = arr[:len(FIELDS)]
        return jsonify({"ok": True, "values": arr.tolist()})
    except Exception as e:
        print(f"Error in /api/prefill: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/select_iol', methods=['POST'])
def api_select_iol():
    try:
        payload = request.get_json(silent=True) or {}
        iol_power = payload.get('iol_power')
        sphere = payload.get('sphere')

        # 目前仅记录选择，具体业务逻辑待定
        app.logger.info('接收到 IOL 度数选择: iol_power=%s, sphere=%s', iol_power, sphere)

        return jsonify({"ok": True, "message": "已记录选择"})
    except Exception as exc:
        app.logger.exception('处理 IOL 度数选择时出错')
        return jsonify({"ok": False, "error": str(exc)}), 500
if __name__ == '__main__':
    os.makedirs('static/html', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
