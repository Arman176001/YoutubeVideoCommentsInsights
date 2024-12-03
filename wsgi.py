import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_environment():
    """Print out environment variables and system information for debugging"""
    print("Current Working Directory:", os.getcwd())
    print("Python Version:", sys.version)
    print("Python Path:", sys.path)
    print("Environment Variables:")
    for key, value in os.environ.items():
        print(f"{key}: {value}")

try:
    from app import create_app
    debug_environment()
    app = create_app()
except Exception as e:
    print(f"Error importing or creating app: {e}")
    print(f"Exception Type: {type(e)}")
    import traceback
    traceback.print_exc()
    app = None

# Vercel serverless function handler
def handler(event, context):
    if app is None:
        return {
            'statusCode': 500,
            'body': 'Failed to initialize Flask application'
        }
    
    # Convert Vercel event to Flask-compatible request
    from flask import Request
    from werkzeug.test import create_environ
    from io import BytesIO

    try:
        environ = create_environ(
            path=event.get('path', '/'),
            method=event.get('httpMethod', 'GET'),
            input_stream=BytesIO(event.get('body', '').encode('utf-8') if event.get('body') else b''),
            headers=event.get('headers', {})
        )
        
        response = app(environ, lambda status, headers: None)
        
        return {
            'statusCode': response.status_code,
            'body': response.get_data(as_text=True),
            'headers': dict(response.headers)
        }
    except Exception as e:
        print(f"Error in handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': f'Internal Server Error: {str(e)}'
        }