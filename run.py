import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

if __name__ == '__main__':
    print("Running on http://localhost:5000 ")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
