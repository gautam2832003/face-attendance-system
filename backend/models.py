from datetime import datetime, timezone


class AdminModel:
    collection = 'admins'

    @staticmethod
    def create(username, password, name, email):
        return {
            'username': username,
            'password': password,
            'name': name,
            'email': email,
            'created_at': datetime.now(timezone.utc)
        }


class EmployeeModel:
    collection = 'employees'

    @staticmethod
    def create(username, password, name, email, employee_code, department, position, salary_rate=0):
        return {
            'username': username,
            'password': password,
            'name': name,
            'email': email,
            'employee_code': employee_code,
            'department': department,
            'position': position,
            'salary_rate': float(salary_rate),
            'face_encoding': None,
            'face_image_path': None,
            'is_active': True,
            'created_at': datetime.now(timezone.utc)
        }


class AttendanceModel:
    collection = 'attendances'

    @staticmethod
    def create(employee_id, check_in_time, date_str, status='present'):
        return {
            'employee_id': employee_id,
            'date': date_str,
            'check_in': check_in_time,
            'check_out': None,
            'status': status,
            'marked_by': 'face',
            'admin_id': None,
            'created_at': datetime.now(timezone.utc)
        }


class TaskModel:
    collection = 'tasks'

    @staticmethod
    def create(assigned_by, assigned_to, title, description, due_date, priority='medium'):
        return {
            'assigned_by': assigned_by,
            'assigned_to': assigned_to,
            'title': title,
            'description': description,
            'due_date': due_date,
            'priority': priority,
            'status': 'pending',
            'created_at': datetime.now(timezone.utc),
            'completed_at': None
        }


class MailModel:
    collection = 'mails'

    @staticmethod
    def create(sender_id, sender_type, receiver_id, receiver_type, subject, message):
        return {
            'sender_id': sender_id,
            'sender_type': sender_type,
            'receiver_id': receiver_id,
            'receiver_type': receiver_type,
            'subject': subject,
            'message': message,
            'is_read': False,
            'created_at': datetime.now(timezone.utc)
        }


class PayrollModel:
    collection = 'payrolls'

    @staticmethod
    def create(employee_id, month, year, salary_amount, bonus=0, deductions=0, remarks=''):
        net = float(salary_amount) + float(bonus) - float(deductions)
        return {
            'employee_id': employee_id,
            'month': int(month),
            'year': int(year),
            'salary_amount': float(salary_amount),
            'bonus': float(bonus),
            'deductions': float(deductions),
            'net_salary': round(net, 2),
            'status': 'unpaid',
            'paid_date': None,
            'remarks': remarks,
            'created_at': datetime.now(timezone.utc)
        }
