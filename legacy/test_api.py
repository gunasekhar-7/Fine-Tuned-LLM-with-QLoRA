import sys
sys.path.insert(0, r'C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\api')
from main import app

print('FastAPI app created successfully')
routes = []
for r in app.routes:
    if hasattr(r, 'methods'):
        routes.append(r.path)
print(f'Routes: {routes}')