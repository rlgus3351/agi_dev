🧩 외부 데이터 메타데이터 관리 테이블 구조

외부에서 수집된 원시 데이터를 주제별 코드로 분류,
수집 메타데이터 관리 → 품질 검증 결과 관리까지 일관되게 추적하기 위한 3단계 구조입니다.

📘 1️⃣ tb_external_code

데이터 코드 마스터 테이블 (주제 기준 테이블)

외부 데이터의 주제(category)와 질환별 분류를 정의하는 기준 코드 테이블입니다.
모든 외부 데이터는 반드시 이 테이블의 code_name을 참조합니다.

컬럼명 타입 설명
code_name varchar(100) 외부 데이터 코드명 (예: P-001, D-002)
category_name varchar(255) 세부 주제명 (예: 파킨슨 증상, 우울증 진단 방법 등)
disease_type varchar(50) 질환 유형 (PD = 파킨슨, MDD = 우울증 등)
description text 코드 설명 (선택)
created_at timestamp 등록 일시

📌 역할

데이터의 분류 체계를 표준화

이후 수집 및 검증 단계의 기준(FK)으로 사용

📦 2️⃣ tb_external_collection

외부 수집 데이터 메타데이터 테이블 (수집 단계)

외부 데이터를 수집할 때 생성되는 기술적 메타데이터를 관리합니다.
(수집 건수, 누락률, 용량, 소요 시간 등)

컬럼명 타입 설명
external_id serial 고유 식별자 (PK)
code_name varchar(100) 외부 데이터 코드명 (FK → tb_external_code)
data_category varchar(50) 데이터 상위 카테고리 (PD, MDD 등)
data_type varchar(50) 데이터 유형 (corpus, qna, video 등)
num_samples integer 수집된 샘플 개수
total_sentences integer 총 문장 수
total_tokens integer 총 토큰 수
reference_missing_rate double precision 참조(reference) 누락률
doi_missing_rate double precision DOI 누락률
verification_missing_rate double precision 검증 항목 누락률
file_size_mb numeric(10,2) 수집된 데이터 파일 크기 (MB 단위)
collected_date timestamp 수집 완료 일시
collection_latency interval 수집 소요 시간
created_at timestamp 등록 일시 (기본값: now)

📌 역할

실제 수집된 데이터의 기본 통계와 품질 지표 관리

수집 시점과 소요 시간, 데이터 유형별 분류 등 기록

tb_external_code를 참조해 주제별 데이터와 연결

💡 비유

“데이터 수집 일지” — 언제, 얼마만큼, 어떤 데이터를 가져왔는지 기록하는 로그

✅ 3️⃣ tb_external_validation

외부 수집 데이터 검증 결과 테이블 (품질 점검 단계)

수집된 데이터의 품질 검증 결과를 저장합니다.
검증 점수, 누락률, PASS/FAIL 결과 등이 포함됩니다.

컬럼명 타입 설명
validation_id serial 검증 결과 식별자 (PK)
code_name varchar(100) 검증 대상 코드명 (FK → tb_external_code)
data_category varchar(50) 데이터 상위 카테고리 (PD, MDD 등)
data_type varchar(50) 검증 대상 데이터 타입 (corpus, qna 등)
validation_type varchar(100) 검증 유형 (예: auto_quality_check 등)
validation_result varchar(50) 검증 결과 (PASS, FAIL, WARNING 등)
validation_score float8 검증 점수 (0.0 ~ 1.0)
verification_missing_rate double precision 검증 항목 누락률
reference_missing_rate double precision 참조 누락률
doi_missing_rate double precision DOI 누락률
file_size_mb numeric(10,2) 검증 산출물 파일 크기 (MB 단위)
collection_latency interval 검증 과정 소요 시간
checked_at timestamp 검증 완료 일시 (collected_date 종료 시간 기준)
reviewer varchar(100) 검증 담당자
notes text 비고사항

📌 역할

수집된 데이터의 품질 검증 및 결과 저장

자동 검증 로직(auto_quality_check)으로 품질 점수 산출

PASS/FAIL 기준을 정량적으로 관리

💡 비유

“품질 검사표” — 데이터가 제대로 수집·정제되었는지 확인하는 단계

🔗 테이블 관계 (ER 다이어그램 개념)
tb_external_code (데이터 분류 기준)
│
├── tb_external_collection (수집 메타데이터)
│
└── tb_external_validation (검증 결과)

데이터 흐름

외부 데이터 수집
↓
수집 메타데이터 기록 (tb_external_collection)
↓
품질 검증 및 점수 계산
↓
검증 결과 기록 (tb_external_validation)

🧾 요약
단계 테이블명 역할
🧩 1단계 tb_external_code 데이터 코드 기준 관리 (무엇을 수집하는가)
📦 2단계 tb_external_collection 수집된 데이터 메타데이터 관리 (어떻게 수집했는가)
✅ 3단계 tb_external_validation 품질 검증 결과 관리 (잘 수집되었는가)

📘 한 줄 요약

tb_external_code는 무엇을,
tb_external_collection은 어떻게,
tb_external_validation은 얼마나 잘 수집되었는가를 관리한다.
