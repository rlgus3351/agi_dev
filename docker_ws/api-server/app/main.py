from fastapi import FastAPI
import database
import schemas  
from routers import patient,item, health,meta
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="FastAPI Patient API",
    description="데이터 입력 프로그램",
    version="1.0.0"
)

# Instrumentator 인스턴스 생성 및 등록
Instrumentator().instrument(app).expose(app)

# ✅ Swagger 문서에 스키마 예시 반영 (선택)
example_schema = schemas.Patient
example_create = schemas.PatientCreate
example_schema = schemas.ItemCreate
example_create = schemas.VideoMeta
example_schema = schemas.VideoMetaCreate

# 라우터 등록
app.include_router(health.router)
app.include_router(patient.router)
app.include_router(item.router)
app.include_router(meta.router)
