import os
import sys
from datetime import datetime, date, time, timedelta, timezone
from bson import ObjectId
from flask import (
    Flask, render_template, request, jsonify, session, redirect, url_for
)
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from functools import wraps
import calendar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.face_recognizer import (
    extract_face_encoding, find_best_match, save_face_image
)
from backend.ml_trainer import (
    train_all_models, load_trained_models, generate_csv_dataset,
    get_model_accuracy_history, predict_with_models
)
from backend.models import (
    AdminModel, EmployeeModel, AttendanceModel, TaskModel, MailModel, PayrollModel
)

load_dotenv()

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
)
app.secret_key = os.getenv('SECRET_KEY', 'face-attendance-secret')
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/face_attendance')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads/faces')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)

CORS(app)
bcrypt = Bcrypt(app)
mongo = PyMongo(app)

db = mongo.db
admins_col = db.admins
employees_col = db.employees
attendance_col = db.attendances
tasks_col = db.tasks
mails_col = db.mails
payrolls_col = db.payrolls

admins_col.create_index('username', unique=True)
admins_col.create_index('email', unique=True)
employees_col.create_index('username', unique=True)
employees_col.create_index('email', unique=True)
employees_col.create_index('employee_code', unique=True, sparse=True)
attendance_col.create_index([('employee_id', 1), ('date', 1)])
tasks_col.create_index([('assigned_to', 1), ('status', 1)])
mails_col.create_index([('receiver_id', 1), ('is_read', 1)])
payrolls_col.create_index([('employee_id', 1), ('month', 1), ('year', 1)])


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session or session.get('user_type') != 'admin':
            if request.is_json:
                return jsonify({'error': 'Admin access required'}), 403
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


def employee_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session or session.get('user_type') != 'employee':
            if request.is_json:
                return jsonify({'error': 'Employee access required'}), 403
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'GET':
        return render_template('admin_register.html')
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    username = data.get('username', '').strip()
    password = data.get('password', '')
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    if not all([username, password, name, email]):
        return jsonify({'error': 'All fields are required'}), 400
    if admins_col.find_one({'username': username}):
        return jsonify({'error': 'Username already exists'}), 409
    if admins_col.find_one({'email': email}):
        return jsonify({'error': 'Email already exists'}), 409
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    admin = AdminModel.create(username, hashed, name, email)
    result = admins_col.insert_one(admin)
    return jsonify({'message': 'Admin registered successfully', 'id': str(result.inserted_id)}), 201


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'GET':
        return render_template('admin_login.html')
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    admin = admins_col.find_one({'username': username})
    if not admin or not bcrypt.check_password_hash(admin['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    session.permanent = True
    session['user_id'] = str(admin['_id'])
    session['user_type'] = 'admin'
    session['user_name'] = admin['name']
    return jsonify({'message': 'Login successful', 'redirect': url_for('admin_dashboard')})


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html', admin_name=session.get('user_name'))


@app.route('/admin/payroll')
@admin_required
def admin_payroll():
    return render_template('admin_payroll.html', admin_name=session.get('user_name'))


@app.route('/admin/ml')
@admin_required
def admin_ml():
    return render_template('admin_ml_dashboard.html', admin_name=session.get('user_name'))


@app.route('/admin/tasks')
@admin_required
def admin_tasks():
    return render_template('admin_tasks.html', admin_name=session.get('user_name'))


@app.route('/admin/mail')
@admin_required
def admin_mail():
    return render_template('admin_mail.html', admin_name=session.get('user_name'))


@app.route('/employee/register', methods=['GET', 'POST'])
def employee_register():
    if request.method == 'GET':
        return render_template('employee_register.html')
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    username = data.get('username', '').strip()
    password = data.get('password', '')
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    employee_code = data.get('employee_code', '').strip()
    department = data.get('department', '').strip()
    position = data.get('position', '').strip()
    if not all([username, password, name, email, employee_code, department, position]):
        return jsonify({'error': 'All fields are required'}), 400
    if employees_col.find_one({'username': username}):
        return jsonify({'error': 'Username already exists'}), 409
    if employees_col.find_one({'email': email}):
        return jsonify({'error': 'Email already exists'}), 409
    if employees_col.find_one({'employee_code': employee_code}):
        return jsonify({'error': 'Employee code already exists'}), 409
    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    employee = EmployeeModel.create(username, hashed, name, email, employee_code, department, position)
    result = employees_col.insert_one(employee)
    return jsonify({'message': 'Employee registered successfully', 'id': str(result.inserted_id)}), 201


@app.route('/employee/login', methods=['GET', 'POST'])
def employee_login():
    if request.method == 'GET':
        return render_template('employee_login.html')
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    employee = employees_col.find_one({'username': username})
    if not employee or not bcrypt.check_password_hash(employee['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    session.permanent = True
    session['user_id'] = str(employee['_id'])
    session['user_type'] = 'employee'
    session['user_name'] = employee['name']
    session['employee_code'] = employee.get('employee_code', '')
    return jsonify({'message': 'Login successful', 'redirect': url_for('employee_dashboard')})


@app.route('/employee/dashboard')
@employee_required
def employee_dashboard():
    return render_template('employee_dashboard.html', employee_name=session.get('user_name'))


@app.route('/employee/tasks')
@employee_required
def employee_tasks():
    return render_template('employee_tasks.html', employee_name=session.get('user_name'))


@app.route('/employee/mail')
@employee_required
def employee_mail():
    return render_template('employee_mail.html', employee_name=session.get('user_name'))


@app.route('/employee/salary')
@employee_required
def employee_salary():
    return render_template('employee_salary.html', employee_name=session.get('user_name'))


# ---- Employee API Routes ----

@app.route('/api/employee/register-face', methods=['POST'])
@employee_required
def register_face():
    employee_id = session['user_id']
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    image_data = data['image']
    encoding_b64, msg = extract_face_encoding(image_data)
    if encoding_b64 is None:
        return jsonify({'error': msg}), 400
    image_path = save_face_image(image_data, employee_id)
    employees_col.update_one(
        {'_id': ObjectId(employee_id)},
        {'$set': {
            'face_encoding': encoding_b64,
            'face_image_path': image_path
        }}
    )
    return jsonify({'message': 'Face registered successfully!'}), 200


@app.route('/api/employee/profile', methods=['GET'])
@employee_required
def employee_profile():
    employee_id = session['user_id']
    emp = employees_col.find_one({'_id': ObjectId(employee_id)})
    if not emp:
        return jsonify({'error': 'Employee not found'}), 404
    return jsonify({
        'name': emp['name'],
        'email': emp['email'],
        'employee_code': emp.get('employee_code', ''),
        'department': emp.get('department', ''),
        'position': emp.get('position', ''),
        'salary_rate': emp.get('salary_rate', 0),
        'has_face': emp.get('face_encoding') is not None
    })


@app.route('/api/employee/attendance/check-in', methods=['POST'])
@employee_required
def employee_check_in():
    employee_id = session['user_id']
    today = date.today().isoformat()
    now = datetime.now(timezone.utc)
    time_str = now.strftime('%H:%M:%S')
    existing = attendance_col.find_one({'employee_id': employee_id, 'date': today})
    if existing:
        if existing.get('check_in'):
            return jsonify({'error': 'Already checked in today', 'check_in': existing.get('check_in')}), 400
    if existing:
        attendance_col.update_one({'_id': existing['_id']}, {'$set': {'check_in': time_str, 'status': 'present', 'marked_by': 'face'}})
    else:
        attendance_col.insert_one({
            'employee_id': employee_id, 'date': today, 'check_in': time_str,
            'check_out': None, 'status': 'present', 'marked_by': 'face',
            'admin_id': None, 'created_at': datetime.now(timezone.utc)
        })
    return jsonify({'message': 'Check-in recorded', 'time': time_str})


@app.route('/api/employee/attendance/check-out', methods=['POST'])
@employee_required
def employee_check_out():
    employee_id = session['user_id']
    today = date.today().isoformat()
    now = datetime.now(timezone.utc)
    time_str = now.strftime('%H:%M:%S')
    existing = attendance_col.find_one({'employee_id': employee_id, 'date': today})
    if not existing:
        return jsonify({'error': 'No check-in found for today'}), 400
    if existing.get('check_out'):
        return jsonify({'error': 'Already checked out today', 'check_out': existing.get('check_out')}), 400
    check_in_time = existing.get('check_in', '00:00:00')
    try:
        ci = datetime.strptime(check_in_time, '%H:%M:%S')
        co = datetime.strptime(time_str, '%H:%M:%S')
        hours_worked = (co - ci).total_seconds() / 3600
    except Exception:
        hours_worked = 0
    status = 'present'
    if hours_worked < 4:
        status = 'half-day'
    attendance_col.update_one(
        {'_id': existing['_id']},
        {'$set': {'check_out': time_str, 'status': status}}
    )
    return jsonify({'message': 'Check-out recorded', 'time': time_str, 'hours_worked': round(hours_worked, 2)})


@app.route('/api/employee/attendance/today', methods=['GET'])
@employee_required
def employee_today_attendance():
    employee_id = session['user_id']
    today = date.today().isoformat()
    record = attendance_col.find_one({'employee_id': employee_id, 'date': today})
    if not record:
        return jsonify({'checked_in': False, 'checked_out': False})
    return jsonify({
        'checked_in': record.get('check_in') is not None,
        'checked_out': record.get('check_out') is not None,
        'check_in': record.get('check_in', ''),
        'check_out': record.get('check_out', ''),
        'status': record.get('status', ''),
        'marked_by': record.get('marked_by', '')
    })


@app.route('/api/employee/attendance/history', methods=['GET'])
@employee_required
def employee_attendance_history():
    employee_id = session['user_id']
    records = list(attendance_col.find({'employee_id': employee_id}).sort('date', -1).limit(60))
    result = []
    for r in records:
        result.append({
            'id': str(r['_id']),
            'date': r.get('date', ''),
            'check_in': r.get('check_in', ''),
            'check_out': r.get('check_out', ''),
            'status': r.get('status', 'present'),
            'marked_by': r.get('marked_by', 'face')
        })
    return jsonify(result)


@app.route('/api/employee/attendance/stats', methods=['GET'])
@employee_required
def employee_attendance_stats():
    employee_id = session['user_id']
    total = attendance_col.count_documents({'employee_id': employee_id})
    present = attendance_col.count_documents({'employee_id': employee_id, 'status': 'present'})
    half_day = attendance_col.count_documents({'employee_id': employee_id, 'status': 'half-day'})
    absent = attendance_col.count_documents({'employee_id': employee_id, 'status': 'absent'})
    percentage = ((present + half_day * 0.5) / total * 100) if total > 0 else 0
    return jsonify({
        'total': total, 'present': present, 'half_day': half_day,
        'absent': absent, 'percentage': round(percentage, 1)
    })


# ---- Employee Tasks ----

@app.route('/api/employee/tasks', methods=['GET'])
@employee_required
def get_employee_tasks():
    employee_id = session['user_id']
    status_filter = request.args.get('status', '')
    query = {'assigned_to': employee_id}
    if status_filter:
        query['status'] = status_filter
    tasks = list(tasks_col.find(query).sort('created_at', -1))
    result = []
    for t in tasks:
        admin = admins_col.find_one({'_id': ObjectId(t['assigned_by'])}) if ObjectId.is_valid(t['assigned_by']) else None
        result.append({
            'id': str(t['_id']),
            'title': t.get('title', ''),
            'description': t.get('description', ''),
            'due_date': t.get('due_date', ''),
            'priority': t.get('priority', 'medium'),
            'status': t.get('status', 'pending'),
            'assigned_by_name': admin.get('name', 'Admin') if admin else 'Admin',
            'created_at': t.get('created_at').isoformat() if t.get('created_at') else ''
        })
    return jsonify(result)


@app.route('/api/employee/tasks/<task_id>/status', methods=['PUT'])
@employee_required
def update_task_status(task_id):
    employee_id = session['user_id']
    data = request.json
    new_status = data.get('status', '')
    if new_status not in ['pending', 'in_progress', 'completed']:
        return jsonify({'error': 'Invalid status'}), 400
    task = tasks_col.find_one({'_id': ObjectId(task_id), 'assigned_to': employee_id})
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    update = {'status': new_status}
    if new_status == 'completed':
        update['completed_at'] = datetime.now(timezone.utc)
    tasks_col.update_one({'_id': ObjectId(task_id)}, {'$set': update})
    return jsonify({'message': 'Task updated'})


# ---- Employee Mail ----

@app.route('/api/employee/mail', methods=['GET'])
@employee_required
def get_employee_mail():
    employee_id = session['user_id']
    mail_type = request.args.get('type', 'inbox')
    if mail_type == 'sent':
        query = {'sender_id': employee_id, 'sender_type': 'employee'}
    else:
        query = {'receiver_id': employee_id, 'receiver_type': 'employee'}
    mails = list(mails_col.find(query).sort('created_at', -1).limit(50))
    result = []
    for m in mails:
        sender_name = 'Unknown'
        if m.get('sender_type') == 'admin':
            admin = admins_col.find_one({'_id': ObjectId(m['sender_id'])}) if ObjectId.is_valid(m['sender_id']) else None
            sender_name = admin.get('name', 'Admin') if admin else 'Admin'
        else:
            emp = employees_col.find_one({'_id': ObjectId(m['sender_id'])}) if ObjectId.is_valid(m['sender_id']) else None
            sender_name = emp.get('name', 'Unknown') if emp else 'Unknown'
        result.append({
            'id': str(m['_id']),
            'subject': m.get('subject', ''),
            'message': m.get('message', ''),
            'sender_name': sender_name,
            'sender_type': m.get('sender_type', ''),
            'is_read': m.get('is_read', False),
            'created_at': m.get('created_at').isoformat() if m.get('created_at') else ''
        })
    return jsonify(result)


@app.route('/api/employee/mail/send', methods=['POST'])
@employee_required
def send_employee_mail():
    employee_id = session['user_id']
    data = request.json
    if not data or not data.get('subject') or not data.get('message'):
        return jsonify({'error': 'Subject and message required'}), 400
    receiver_id = data.get('receiver_id', '')
    if not receiver_id:
        admins = list(admins_col.find())
        if not admins:
            return jsonify({'error': 'No admin available'}), 400
        receiver_id = str(admins[0]['_id'])
    mail_data = MailModel.create(
        employee_id, 'employee', receiver_id, 'admin',
        data['subject'], data['message']
    )
    mails_col.insert_one(mail_data)
    return jsonify({'message': 'Mail sent successfully'}), 201


@app.route('/api/employee/mail/<mail_id>/read', methods=['PUT'])
@employee_required
def mark_mail_read(mail_id):
    employee_id = session['user_id']
    mails_col.update_one(
        {'_id': ObjectId(mail_id), 'receiver_id': employee_id},
        {'$set': {'is_read': True}}
    )
    return jsonify({'message': 'Marked as read'})


@app.route('/api/employee/mail/unread-count', methods=['GET'])
@employee_required
def employee_unread_count():
    employee_id = session['user_id']
    count = mails_col.count_documents({'receiver_id': employee_id, 'receiver_type': 'employee', 'is_read': False})
    return jsonify({'unread': count})


# ---- Employee Salary ----

@app.route('/api/employee/salary', methods=['GET'])
@employee_required
def get_employee_salary():
    employee_id = session['user_id']
    emp = employees_col.find_one({'_id': ObjectId(employee_id)})
    if not emp:
        return jsonify({'error': 'Not found'}), 404
    payrolls = list(payrolls_col.find({'employee_id': employee_id}).sort([('year', -1), ('month', -1)]))
    history = [{
        'month': p['month'], 'year': p['year'],
        'salary_amount': p.get('salary_amount', 0),
        'bonus': p.get('bonus', 0), 'deductions': p.get('deductions', 0),
        'net_salary': p.get('net_salary', 0), 'status': p.get('status', 'unpaid'),
        'paid_date': p.get('paid_date', '')
    } for p in payrolls]
    return jsonify({
        'salary_rate': emp.get('salary_rate', 0),
        'department': emp.get('department', ''),
        'position': emp.get('position', ''),
        'history': history
    })


# ---- Admin API Routes ----

@app.route('/api/admin/employees', methods=['GET'])
@admin_required
def get_employees():
    employees = list(employees_col.find())
    result = []
    for e in employees:
        result.append({
            'id': str(e['_id']),
            'name': e['name'], 'email': e['email'],
            'employee_code': e.get('employee_code', ''),
            'department': e.get('department', ''),
            'position': e.get('position', ''),
            'salary_rate': e.get('salary_rate', 0),
            'has_face': e.get('face_encoding') is not None,
            'is_active': e.get('is_active', True)
        })
    return jsonify(result)


@app.route('/api/admin/employees/<emp_id>/salary', methods=['PUT'])
@admin_required
def update_employee_salary(emp_id):
    data = request.json
    salary_rate = data.get('salary_rate')
    if salary_rate is None:
        return jsonify({'error': 'salary_rate required'}), 400
    employees_col.update_one(
        {'_id': ObjectId(emp_id)},
        {'$set': {'salary_rate': float(salary_rate)}}
    )
    return jsonify({'message': 'Salary updated'})


@app.route('/api/admin/attendance/today', methods=['GET'])
@admin_required
def admin_today_attendance():
    today = date.today().isoformat()
    records = list(attendance_col.find({'date': today}))
    emp_ids = list(set(r['employee_id'] for r in records))
    emp_map = {}
    if emp_ids:
        oids = [ObjectId(eid) for eid in emp_ids if ObjectId.is_valid(eid)]
        for e in employees_col.find({'_id': {'$in': oids}}):
            emp_map[str(e['_id'])] = e
    result = []
    for r in records:
        eid = r['employee_id']
        emp = emp_map.get(eid, {})
        result.append({
            'id': str(r['_id']), 'employee_id': eid,
            'employee_name': emp.get('name', 'Unknown'),
            'employee_code': emp.get('employee_code', ''),
            'check_in': r.get('check_in', ''),
            'check_out': r.get('check_out', ''),
            'status': r.get('status', 'present'),
            'marked_by': r.get('marked_by', 'face')
        })
    return jsonify(result)


@app.route('/api/admin/attendance/summary', methods=['GET'])
@admin_required
def admin_attendance_summary():
    today = date.today().isoformat()
    all_employees = list(employees_col.find({'is_active': True}))
    today_records = list(attendance_col.find({'date': today}))
    marked_map = {r['employee_id']: r for r in today_records}
    result = []
    for e in all_employees:
        eid = str(e['_id'])
        if eid in marked_map:
            r = marked_map[eid]
            result.append({
                'id': eid, 'name': e['name'],
                'employee_code': e.get('employee_code', ''),
                'department': e.get('department', ''),
                'check_in': r.get('check_in', ''),
                'check_out': r.get('check_out', ''),
                'status': r.get('status', 'unmarked'),
                'marked': True
            })
        else:
            result.append({
                'id': eid, 'name': e['name'],
                'employee_code': e.get('employee_code', ''),
                'department': e.get('department', ''),
                'check_in': '', 'check_out': '',
                'status': 'unmarked', 'marked': False
            })
    present = sum(1 for r in result if r['status'] == 'present')
    absent = sum(1 for r in result if r['status'] == 'absent')
    half_day = sum(1 for r in result if r['status'] == 'half-day')
    unmarked = sum(1 for r in result if r['status'] == 'unmarked')
    return jsonify({
        'employees': result,
        'stats': {'total': len(result), 'present': present, 'absent': absent, 'half_day': half_day, 'unmarked': unmarked}
    })


@app.route('/api/admin/attendance/recognize', methods=['POST'])
@admin_required
def admin_recognize_face():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    image_data = data['image']
    encoding_b64, msg = extract_face_encoding(image_data)
    if encoding_b64 is None:
        return jsonify({'error': msg}), 400
    employees = list(employees_col.find({'face_encoding': {'$ne': None}, 'is_active': True}))
    if not employees:
        return jsonify({'error': 'No registered employees found'}), 404
    match, distance = find_best_match(encoding_b64, employees, tolerance=0.5)
    if match is None:
        return jsonify({'matched': False, 'error': 'No matching employee found', 'distance': float(distance)})
    employee_id = str(match['_id'])
    today = date.today().isoformat()
    now = datetime.now(timezone.utc)
    time_str = now.strftime('%H:%M:%S')
    existing = attendance_col.find_one({'employee_id': employee_id, 'date': today})
    if existing:
        if not existing.get('check_in'):
            attendance_col.update_one({'_id': existing['_id']}, {'$set': {'check_in': time_str, 'marked_by': 'face', 'admin_id': session['user_id']}})
            return jsonify({
                'matched': True, 'action': 'check_in',
                'employee': {'id': employee_id, 'name': match['name'], 'employee_code': match.get('employee_code', ''), 'department': match.get('department', '')},
                'time': time_str, 'distance': float(distance)
            })
        elif not existing.get('check_out'):
            attendance_col.update_one({'_id': existing['_id']}, {'$set': {'check_out': time_str, 'admin_id': session['user_id']}})
            return jsonify({
                'matched': True, 'action': 'check_out',
                'employee': {'id': employee_id, 'name': match['name'], 'employee_code': match.get('employee_code', ''), 'department': match.get('department', '')},
                'time': time_str, 'distance': float(distance)
            })
        else:
            return jsonify({
                'matched': True, 'action': 'already_completed',
                'employee': {'id': employee_id, 'name': match['name'], 'employee_code': match.get('employee_code', ''), 'department': match.get('department', '')},
                'check_in': existing.get('check_in'), 'check_out': existing.get('check_out'),
                'distance': float(distance)
            })
    attendance_col.insert_one({
        'employee_id': employee_id, 'date': today, 'check_in': time_str,
        'check_out': None, 'status': 'present', 'marked_by': 'face',
        'admin_id': session['user_id'], 'created_at': datetime.now(timezone.utc)
    })
    return jsonify({
        'matched': True, 'action': 'check_in',
        'employee': {'id': employee_id, 'name': match['name'], 'employee_code': match.get('employee_code', ''), 'department': match.get('department', '')},
        'time': time_str, 'distance': float(distance)
    })


@app.route('/api/admin/attendance/date-range', methods=['POST'])
@admin_required
def admin_attendance_date_range():
    data = request.json
    start = data.get('start_date', date.today().isoformat())
    end = data.get('end_date', date.today().isoformat())
    records = list(attendance_col.find({'date': {'$gte': start, '$lte': end}}).sort('date', -1))
    emp_ids = list(set(r['employee_id'] for r in records))
    emp_map = {}
    if emp_ids:
        oids = [ObjectId(eid) for eid in emp_ids if ObjectId.is_valid(eid)]
        for e in employees_col.find({'_id': {'$in': oids}}):
            emp_map[str(e['_id'])] = e
    result = []
    for r in records:
        eid = r['employee_id']
        emp = emp_map.get(eid, {})
        result.append({
            'date': r.get('date', ''), 'check_in': r.get('check_in', ''),
            'check_out': r.get('check_out', ''),
            'employee_name': emp.get('name', 'Unknown'),
            'employee_code': emp.get('employee_code', ''),
            'department': emp.get('department', ''),
            'status': r.get('status', 'present')
        })
    return jsonify(result)


@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    total_employees = employees_col.count_documents({'is_active': True})
    today = date.today().isoformat()
    today_present = attendance_col.count_documents({'date': today, 'status': 'present'})
    today_absent = attendance_col.count_documents({'date': today, 'status': 'absent'})
    today_half = attendance_col.count_documents({'date': today, 'status': 'half-day'})
    with_face = employees_col.count_documents({'face_encoding': {'$ne': None}})
    pending_tasks = tasks_col.count_documents({'status': {'$ne': 'completed'}})
    unread_mails = mails_col.count_documents({'receiver_type': 'admin', 'is_read': False})
    return jsonify({
        'total_employees': total_employees, 'today_present': today_present,
        'today_absent': today_absent, 'today_half_day': today_half,
        'face_registered': with_face, 'pending_tasks': pending_tasks,
        'unread_mails': unread_mails
    })


@app.route('/api/admin/attendance/mark-manual', methods=['POST'])
@admin_required
def admin_mark_manual():
    data = request.json
    employee_id = data.get('employee_id', '')
    status = data.get('status', '')
    if not employee_id or not status:
        return jsonify({'error': 'employee_id and status required'}), 400
    if status not in ['present', 'absent', 'half-day']:
        return jsonify({'error': 'Invalid status'}), 400
    today = date.today().isoformat()
    now = datetime.now(timezone.utc)
    time_str = now.strftime('%H:%M:%S')
    existing = attendance_col.find_one({'employee_id': employee_id, 'date': today})
    if existing:
        attendance_col.update_one(
            {'_id': existing['_id']},
            {'$set': {'status': status, 'check_in': existing.get('check_in') or time_str, 'marked_by': 'admin', 'admin_id': session['user_id']}}
        )
        return jsonify({'message': 'Attendance updated'})
    attendance_col.insert_one({
        'employee_id': employee_id, 'date': today, 'check_in': time_str,
        'check_out': None, 'status': status, 'marked_by': 'admin',
        'admin_id': session['user_id'], 'created_at': datetime.now(timezone.utc)
    })
    return jsonify({'message': 'Attendance marked'})


# ---- Admin Payroll ----

@app.route('/api/admin/payroll', methods=['GET'])
@admin_required
def get_payrolls():
    month = request.args.get('month', '')
    year = request.args.get('year', '')
    query = {}
    if month:
        query['month'] = int(month)
    if year:
        query['year'] = int(year)
    payrolls = list(payrolls_col.find(query).sort([('year', -1), ('month', -1)]))
    emp_ids = list(set(p['employee_id'] for p in payrolls))
    emp_map = {}
    if emp_ids:
        oids = [ObjectId(eid) for eid in emp_ids if ObjectId.is_valid(eid)]
        for e in employees_col.find({'_id': {'$in': oids}}):
            emp_map[str(e['_id'])] = e
    result = []
    for p in payrolls:
        eid = p['employee_id']
        emp = emp_map.get(eid, {})
        result.append({
            'id': str(p['_id']), 'employee_id': eid,
            'employee_name': emp.get('name', 'Unknown'),
            'employee_code': emp.get('employee_code', ''),
            'department': emp.get('department', ''),
            'month': p['month'], 'year': p['year'],
            'salary_amount': p.get('salary_amount', 0),
            'bonus': p.get('bonus', 0), 'deductions': p.get('deductions', 0),
            'net_salary': p.get('net_salary', 0), 'status': p.get('status', 'unpaid'),
            'paid_date': p.get('paid_date', ''), 'remarks': p.get('remarks', '')
        })
    return jsonify(result)


@app.route('/api/admin/payroll/generate', methods=['POST'])
@admin_required
def generate_payroll():
    data = request.json
    month = int(data.get('month', datetime.now().month))
    year = int(data.get('year', datetime.now().year))
    employees = list(employees_col.find({'is_active': True}))
    created = 0
    for emp in employees:
        eid = str(emp['_id'])
        existing = payrolls_col.find_one({'employee_id': eid, 'month': month, 'year': year})
        if existing:
            continue
        salary_rate = emp.get('salary_rate', 0)
        if salary_rate <= 0:
            continue
        payroll_data = PayrollModel.create(eid, month, year, salary_rate)
        payrolls_col.insert_one(payroll_data)
        created += 1
    return jsonify({'message': f'Generated {created} payroll records'})


@app.route('/api/admin/payroll/<payroll_id>/pay', methods=['PUT'])
@admin_required
def pay_payroll(payroll_id):
    data = request.json
    payrolls_col.update_one(
        {'_id': ObjectId(payroll_id)},
        {'$set': {
            'status': 'paid',
            'paid_date': datetime.now(timezone.utc).isoformat(),
            'bonus': float(data.get('bonus', 0)),
            'deductions': float(data.get('deductions', 0)),
            'remarks': data.get('remarks', '')
        }}
    )
    return jsonify({'message': 'Payroll marked as paid'})


@app.route('/api/admin/payroll/<payroll_id>', methods=['DELETE'])
@admin_required
def delete_payroll(payroll_id):
    payrolls_col.delete_one({'_id': ObjectId(payroll_id)})
    return jsonify({'message': 'Payroll deleted'})


# ---- Admin Tasks ----

@app.route('/api/admin/tasks', methods=['GET'])
@admin_required
def get_all_tasks():
    tasks = list(tasks_col.find().sort('created_at', -1))
    emp_ids = list(set(t['assigned_to'] for t in tasks))
    emp_map = {}
    if emp_ids:
        oids = [ObjectId(eid) for eid in emp_ids if ObjectId.is_valid(eid)]
        for e in employees_col.find({'_id': {'$in': oids}}):
            emp_map[str(e['_id'])] = e
    result = []
    for t in tasks:
        eid = t['assigned_to']
        emp = emp_map.get(eid, {})
        result.append({
            'id': str(t['_id']), 'title': t.get('title', ''),
            'description': t.get('description', ''),
            'due_date': t.get('due_date', ''),
            'priority': t.get('priority', 'medium'),
            'status': t.get('status', 'pending'),
            'assigned_to_name': emp.get('name', 'Unknown'),
            'assigned_to_code': emp.get('employee_code', ''),
            'created_at': t.get('created_at').isoformat() if t.get('created_at') else ''
        })
    return jsonify(result)


@app.route('/api/admin/tasks/create', methods=['POST'])
@admin_required
def create_task():
    admin_id = session['user_id']
    data = request.json
    if not data or not data.get('title') or not data.get('assigned_to'):
        return jsonify({'error': 'Title and assigned_to required'}), 400
    task_data = TaskModel.create(
        admin_id, data['assigned_to'], data['title'],
        data.get('description', ''), data.get('due_date', ''),
        data.get('priority', 'medium')
    )
    result = tasks_col.insert_one(task_data)
    return jsonify({'message': 'Task created', 'id': str(result.inserted_id)}), 201


# ---- Admin Mail ----

@app.route('/api/admin/mail', methods=['GET'])
@admin_required
def get_admin_mail():
    mail_type = request.args.get('type', 'inbox')
    if mail_type == 'sent':
        query = {'sender_id': session['user_id'], 'sender_type': 'admin'}
    else:
        query = {'receiver_type': 'admin'}
    mails = list(mails_col.find(query).sort('created_at', -1).limit(50))
    result = []
    for m in mails:
        sender_name = 'Unknown'
        if m.get('sender_type') == 'employee':
            emp = employees_col.find_one({'_id': ObjectId(m['sender_id'])}) if ObjectId.is_valid(m['sender_id']) else None
            sender_name = emp.get('name', 'Unknown') if emp else 'Unknown'
        else:
            sender_name = session.get('user_name', 'Admin')
        result.append({
            'id': str(m['_id']), 'subject': m.get('subject', ''),
            'message': m.get('message', ''),
            'sender_name': sender_name,
            'sender_type': m.get('sender_type', ''),
            'receiver_type': m.get('receiver_type', ''),
            'is_read': m.get('is_read', False),
            'created_at': m.get('created_at').isoformat() if m.get('created_at') else ''
        })
    return jsonify(result)


@app.route('/api/admin/mail/send', methods=['POST'])
@admin_required
def send_admin_mail():
    admin_id = session['user_id']
    data = request.json
    if not data or not data.get('subject') or not data.get('message') or not data.get('receiver_id'):
        return jsonify({'error': 'Subject, message, and receiver_id required'}), 400
    mail_data = MailModel.create(
        admin_id, 'admin', data['receiver_id'], 'employee',
        data['subject'], data['message']
    )
    mails_col.insert_one(mail_data)
    return jsonify({'message': 'Mail sent'}), 201


@app.route('/api/admin/mail/unread-count', methods=['GET'])
@admin_required
def admin_unread_count():
    count = mails_col.count_documents({'receiver_type': 'admin', 'is_read': False})
    return jsonify({'unread': count})


# ---- Admin ML Routes ----

@app.route('/api/admin/ml/dataset', methods=['GET'])
@admin_required
def ml_get_dataset():
    employees = list(employees_col.find({'face_encoding': {'$ne': None}}))
    csv_path = generate_csv_dataset(employees)
    if csv_path and os.path.exists(csv_path):
        return jsonify({
            'dataset_path': csv_path,
            'dataset_exists': True,
            'rows': len(employees),
            'download_url': f'/api/admin/ml/dataset/download/{os.path.basename(csv_path)}'
        })
    return jsonify({'dataset_exists': False, 'message': 'No face data available. Register employee faces first.'})


@app.route('/api/admin/ml/datasets/list', methods=['GET'])
@admin_required
def ml_list_datasets():
    import glob
    datasets = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'face_dataset_*.csv'))
    result = []
    for d in sorted(datasets, reverse=True):
        size = os.path.getsize(d)
        result.append({
            'path': os.path.basename(d),
            'size_kb': round(size / 1024, 2),
            'created': datetime.fromtimestamp(os.path.getctime(d)).isoformat()
        })
    return jsonify(result)


@app.route('/api/admin/ml/dataset/download/<filename>', methods=['GET'])
@admin_required
def ml_download_dataset(filename):
    from flask import send_from_directory
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'),
        filename,
        as_attachment=True
    )


@app.route('/api/admin/ml/train', methods=['POST'])
@admin_required
def ml_train():
    employees = list(employees_col.find({'face_encoding': {'$ne': None}}))
    if len(employees) < 2:
        return jsonify({'error': 'Need at least 2 employees with registered face data'}), 400
    result = train_all_models(employees)
    if 'error' in result:
        return jsonify({'error': result['error']}), 400
    return jsonify(result)


@app.route('/api/admin/ml/status', methods=['GET'])
@admin_required
def ml_status():
    meta = get_model_accuracy_history()
    models, le, _ = load_trained_models()
    if meta and 'results' in meta:
        return jsonify({
            'trained': True,
            'trained_at': meta.get('trained_at', ''),
            'best_model': meta.get('best_model', ''),
            'n_classes': meta.get('n_classes', 0),
            'n_samples': meta.get('n_samples_total', 0),
            'original_samples': meta.get('original_samples', 0),
            'results': meta['results'],
            'dataset_path': meta.get('dataset_path', '')
        })
    return jsonify({'trained': False})


@app.route('/api/admin/ml/employees-with-face', methods=['GET'])
@admin_required
def ml_employees_with_face():
    employees = list(employees_col.find({'face_encoding': {'$ne': None}}))
    result = []
    for e in employees:
        result.append({
            'id': str(e['_id']), 'name': e['name'],
            'employee_code': e.get('employee_code', ''),
            'department': e.get('department', '')
        })
    return jsonify(result)


# ---- Logout ----

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
