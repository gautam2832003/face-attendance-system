import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app, bcrypt, db

with app.app_context():
    admins_col = db.admins
    employees_col = db.employees
    attendances_col = db.attendances
    tasks_col = db.tasks
    mails_col = db.mails
    payrolls_col = db.payrolls

    if admins_col.count_documents({}) > 0:
        print("Database already has data. Skipping seed.")
        sys.exit(0)

    print("Seeding database with enterprise sample data...")

    from datetime import datetime, timedelta, timezone

    admin_specs = [
        ('admin1', 'admin123', 'Gautam Sharma', 'gautam@company.com'),
        ('admin2', 'admin123', 'Priya Patel', 'priya@company.com'),
    ]
    admin_docs = []
    for spec in admin_specs:
        admin_docs.append({
            'username': spec[0],
            'password': bcrypt.generate_password_hash(spec[1]).decode('utf-8'),
            'name': spec[2],
            'email': spec[3],
            'created_at': datetime.now(timezone.utc)
        })
    admin_result = admins_col.insert_many(admin_docs)
    admin_ids = [str(_id) for _id in admin_result.inserted_ids]
    print(f"Created {len(admin_specs)} admins")

    from backend.models import EmployeeModel
    emp_specs = [
        ('emp1', 'password123', 'Rahul Verma', 'rahul@company.com', 'EMP001', 'Engineering', 'Software Engineer', 75000),
        ('emp2', 'password123', 'Sneha Kapoor', 'sneha@company.com', 'EMP002', 'Engineering', 'Senior Developer', 95000),
        ('emp3', 'password123', 'Amit Singh', 'amit@company.com', 'EMP003', 'Marketing', 'Marketing Lead', 65000),
        ('emp4', 'password123', 'Neha Gupta', 'neha@company.com', 'EMP004', 'HR', 'HR Manager', 70000),
        ('emp5', 'password123', 'Vikram Joshi', 'vikram@company.com', 'EMP005', 'Finance', 'Accountant', 60000),
        ('emp6', 'password123', 'Ananya Reddy', 'ananya@company.com', 'EMP006', 'Engineering', 'Junior Developer', 55000),
    ]
    emp_ids = []
    for spec in emp_specs:
        hashed = bcrypt.generate_password_hash(spec[1]).decode('utf-8')
        emp = EmployeeModel.create(spec[0], hashed, spec[2], spec[3], spec[4], spec[5], spec[6], spec[7])
        result = employees_col.insert_one(emp)
        emp_ids.append(str(result.inserted_id))
    print(f"Created {len(emp_specs)} employees")

    today = datetime.now(timezone.utc)
    now_month = today.month

    attendance_records = []
    statuses = ['present', 'present', 'present', 'absent', 'present', 'half-day']
    for i, eid in enumerate(emp_ids):
        for day_offset in range(1, 8):
            d = (today - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            s = statuses[(i + day_offset) % len(statuses)]
            attendance_records.append({
                'employee_id': eid,
                'date': d,
                'check_in': f'{8 + (i % 3):02d}:{(i * 7) % 60:02d}:00',
                'check_out': f'{17 + (i % 2):02d}:{(i * 11) % 60:02d}:00' if s != 'absent' else None,
                'status': s,
                'marked_by': 'face' if s != 'absent' else 'admin',
                'admin_id': None if s != 'absent' else admin_ids[0],
                'created_at': datetime.now(timezone.utc)
            })
    attendances_col.insert_many(attendance_records)
    print(f"Created {len(attendance_records)} attendance records")

    task_records = [
        {'assigned_by': admin_ids[0], 'assigned_to': emp_ids[0], 'title': 'Implement login API', 'description': 'Build REST API for user authentication', 'due_date': (today + timedelta(days=3)).strftime('%Y-%m-%d'), 'priority': 'high', 'status': 'in_progress', 'created_at': datetime.now(timezone.utc), 'completed_at': None},
        {'assigned_by': admin_ids[0], 'assigned_to': emp_ids[1], 'title': 'Code review sprint 5', 'description': 'Review all PRs for sprint 5 delivery', 'due_date': (today + timedelta(days=1)).strftime('%Y-%m-%d'), 'priority': 'high', 'status': 'pending', 'created_at': datetime.now(timezone.utc), 'completed_at': None},
        {'assigned_by': admin_ids[1], 'assigned_to': emp_ids[2], 'title': 'Social media campaign', 'description': 'Plan Q2 social media campaign strategy', 'due_date': (today + timedelta(days=7)).strftime('%Y-%m-%d'), 'priority': 'medium', 'status': 'pending', 'created_at': datetime.now(timezone.utc), 'completed_at': None},
        {'assigned_by': admin_ids[0], 'assigned_to': emp_ids[3], 'title': 'Update employee handbook', 'description': 'Revise HR policies for 2026', 'due_date': (today + timedelta(days=14)).strftime('%Y-%m-%d'), 'priority': 'low', 'status': 'pending', 'created_at': datetime.now(timezone.utc), 'completed_at': None},
        {'assigned_by': admin_ids[1], 'assigned_to': emp_ids[4], 'title': 'Monthly financial report', 'description': 'Prepare end-of-month financial summary', 'due_date': (today + timedelta(days=2)).strftime('%Y-%m-%d'), 'priority': 'high', 'status': 'in_progress', 'created_at': datetime.now(timezone.utc), 'completed_at': None},
    ]
    tasks_col.insert_many(task_records)
    print(f"Created {len(task_records)} tasks")

    mail_records = [
        {'sender_id': admin_ids[0], 'sender_type': 'admin', 'receiver_id': emp_ids[0], 'receiver_type': 'employee', 'subject': 'Welcome to the team!', 'message': 'Welcome Rahul! Check your dashboard for onboarding tasks.', 'is_read': False, 'created_at': datetime.now(timezone.utc)},
        {'sender_id': emp_ids[0], 'sender_type': 'employee', 'receiver_id': admin_ids[0], 'receiver_type': 'admin', 'subject': 'Login API progress', 'message': 'The login API is 80% complete. Will finish by EOD.', 'is_read': False, 'created_at': datetime.now(timezone.utc)},
        {'sender_id': admin_ids[1], 'sender_type': 'admin', 'receiver_id': emp_ids[2], 'receiver_type': 'employee', 'subject': 'Campaign materials', 'message': 'Please find the attached brand guidelines for the campaign.', 'is_read': False, 'created_at': datetime.now(timezone.utc)},
    ]
    mails_col.insert_many(mail_records)
    print(f"Created {len(mail_records)} mail messages")

    from backend.models import PayrollModel
    payroll_records = []
    for i, eid in enumerate(emp_ids):
        salary = emp_specs[i][7]
        for m in range(1, now_month):
            p = PayrollModel.create(eid, m, today.year, salary, bonus=0, deductions=0)
            p['status'] = 'paid'
            p['paid_date'] = datetime.now(timezone.utc).isoformat()
            payroll_records.append(p)
    payrolls_col.insert_many(payroll_records)
    print(f"Created {len(payroll_records)} payroll records")

    print("\n[OK] Enterprise seed completed successfully!")
    print("\n Login Credentials:")
    print("   Admin:    admin1 / admin123")
    print("   Employee: emp1 / password123")
    print("\n ML Training Ready: Register faces for employees, then train models from ML dashboard.")
