
from app import create_app

app = create_app()

def handler(event, context):
    return app(event, context)



    
    