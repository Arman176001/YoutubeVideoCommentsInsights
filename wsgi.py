from app import create_app

app = create_app()

# This is the entry point for Vercel
def handler(event, context):
    return app(event, context)