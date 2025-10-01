네, 지금까지 논의한 내용을 바탕으로 환자 정보와 데이터 검증을 위한 최종 테이블 설계안을 정리하여 보내드립니다.

이 세 테이블은 1:N 관계로 연결되어 환자, 개별 수집 데이터, 그리고 검증 내역을 체계적으로 관리할 수 있습니다.

1. 환자 정보 테이블
테이블명: tb_Patient_Info

설명: 각 환자의 기본 정보(이니셜, 생년월일 등)를 담는 중심 테이블입니다.

SQL

CREATE TABLE tb_Patient_Info (
    patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), -- 환자 고유 식별자 (UUID)
    patient_initials VARCHAR(10),                           -- 환자 이니셜
    birth_date DATE,                                        -- 생년월일
    institution VARCHAR(100),                               -- 수집 기관
    is_data_complete BOOLEAN DEFAULT FALSE,                 -- 데이터 수집 완료 여부
    completion_date TIMESTAMP,                              -- 데이터 수집 완료 시점
    created_at TIMESTAMP                                    -- 환자 정보 최초 저장 시점
);
2. 임상 데이터 수집 항목 테이블
테이블명: tb_Clinical_Data_Items

설명: 환자로부터 수집된 모든 개별 데이터 항목(설문지, 영상 등)을 저장하며, patient_id를 통해 tb_Patient_Info와 연결됩니다.

SQL

CREATE TABLE tb_Items (
    item_id INT AUTO_INCREMENT PRIMARY KEY,          -- 각 수집 항목의 고유 식별자
    patient_id UUID,                                  -- 외래 키: 환자 ID
    data_category VARCHAR(50),                       -- 데이터 분류 (예: 우울증, 파킨슨병)
    data_type VARCHAR(50),                           -- 데이터 유형 (예: 설문지, 영상, 생체 데이터)
    seq INT,                                         -- 동일 유형 내의 순번
    collected_at DATETIME,                           -- 데이터 수집 시점
    description TEXT,                                -- 항목 상세 설명
    FOREIGN KEY (patient_id) REFERENCES tb_Patient_Info(patient_id)
);


3. 데이터 검증 테이블
테이블명: tb_Data_Validation

설명: 각 데이터 항목(item_id)에 대한 검증 내역을 저장하며, item_id를 통해 tb_Clinical_Data_Items와 연결됩니다.

SQL

CREATE TABLE tb_Data_Validation (
    validation_id INT AUTO_INCREMENT PRIMARY KEY,     -- 검증 내역 고유 식별자 (기본 키)
    item_id INT,                                      -- 외래 키: 어떤 항목에 대한 검증인지 연결
    validation_method VARCHAR(50),                    -- 검증 방법
    validation_description TEXT,                      -- 검증에 대한 상세 설명
    validation_datetime DATETIME,                     -- 검증이 이루어진 시점
    FOREIGN KEY (item_id) REFERENCES tb_Clinical_Data_Items(item_id)
);

// ----비디오 
CREATE TABLE tb_Video_Metadata (
    video_metadata_id INT AUTO_INCREMENT PRIMARY KEY,
    item_id INT,                                     -- 외래 키: 어떤 항목에 대한 영상인지 연결
    file_path VARCHAR(255),                          -- 실제 영상 파일의 경로
    file_size_mb DECIMAL(10, 2),                     -- 파일 크기 (MB)
    duration_seconds INT,                            -- 영상 길이 (초)
    resolution VARCHAR(20),                          -- 해상도 (예: '1920x1080')
    frame_rate INT,                                  -- 프레임 레이트 (fps)
    is_anonymized TINYINT(1),                        -- 비식별화 여부 (0:N, 1:Y)
    FOREIGN KEY (item_id) REFERENCES tb_Items(item_id)
);


4.데이터 전처리/비식별화 테이블
테이블명: tb_Data_Preprocessing

설명: tb_Clinical_Data_Items 테이블의 각 항목에 대해 어떤 전처리 작업이 수행되었는지 상세히 기록합니다.

SQL

CREATE TABLE tb_Data_Preprocessing (
    preprocessing_id INT AUTO_INCREMENT PRIMARY KEY,     -- 전처리 작업의 고유 식별자
    item_id INT,                                         -- 외래 키: 어떤 데이터 항목에 대한 작업인지 연결
    data_category VARCHAR(50),                           -- 데이터 분류 (예: 임상 데이터 공통, 우울증)
    preprocessing_type VARCHAR(100),                     -- 전처리 유형 (예: 영상 비식별화, 메타데이터 정규화)
    preprocessing_datetime DATETIME,                     -- 전처리 작업 수행 시작 시점
    processing_duration DECIMAL(10, 2),                  -- 작업 소요 시간 (단위: 초 또는 분)
    description TEXT,                                    -- 작업에 대한 상세 설명
    FOREIGN KEY (item_id) REFERENCES tb_Clinical_Data_Items(item_id)
);





네, 물론입니다. 보내주신 '데이터 조회' 이미지에 있는 모든 항목은 현재 설계된 데이터베이스를 통해 완벽하게 조회할 수 있습니다.

데이터베이스의 **JOIN**과 WHERE 필터링을 활용하면 원하는 조건에 따라 다양한 방식으로 데이터를 통합하고 검색할 수 있습니다.

아래는 각 조회 항목에 대한 SQL 쿼리 예시입니다.

1. 질환 구분 (데이터조회_질환구분)
전체 데이터에서 어떤 질환들이 존재하는지 중복 없이 확인하는 쿼리입니다.

SQL

SELECT DISTINCT data_category
FROM tb_Clinical_Data_Items;
결과 예시: 우울증, 파킨슨병, 외부 데이터

2. 환자 목록 (데이터조회_환자목록)
특정 질환(예: '우울증')을 가진 환자들의 목록을 조회하는 쿼리입니다.

SQL

SELECT DISTINCT T1.patient_id, T1.initials
FROM tb_Patient_Info AS T1
JOIN tb_Clinical_Data_Items AS T2
ON T1.patient_id = T2.patient_id
WHERE T2.data_category = '우울증';
결과 예시: (ID: 1, 이니셜: KDH), (ID: 3, 이니셜: EYS) 등

3. 환자 상세 (데이터조회_환자상세정보)
특정 환자(예: patient_id가 1인 환자)에 대한 상세 데이터를 모두 조회하는 쿼리입니다. 이 쿼리는 '비식별화 영상'처럼 전처리 여부를 확인하는 것도 포함합니다.

SQL

SELECT
    T1.data_type,                     -- 설문지, 영상, 생체데이터 등
    T1.item_number,                   -- 항목 번호
    T1.description AS item_description,  -- 항목 상세 설명
    T2.preprocessing_type,            -- 전처리 유형 (예: 얼굴 식별자 비식별)
    T2.description AS processing_description -- 전처리 상세 설명

FROM
    tb_Clinical_Data_Items AS T1
LEFT JOIN
    tb_Data_Preprocessing AS T2
ON
    T1.item_id = T2.item_id
WHERE
    T1.patient_id = 1
ORDER BY
    T1.data_type, T1.item_number;
쿼리 설명:

LEFT JOIN: tb_Data_Preprocessing 테이블에 전처리 내역이 없더라도, 모든 데이터 항목(tb_Clinical_Data_Items)을 보여주기 위해 LEFT JOIN을 사용합니다.

T2.preprocessing_type이 '얼굴 식별자 비식별'이나 '운동성 검사 비식별'과 같은 값을 가지는지 확인하여, 해당 영상이 비식별화되었는지 판단할 수 있습니다.

이처럼, 설계된 테이블 간의 관계를 활용하면 다양한 조건과 깊이의 데이터 조회가 모두 가능해집니다.


네, 각 질환에 대한 데이터 통계 역시 간단한 쿼리 하나로 모두 조회할 수 있습니다.

이전에 제안해 드린 vw_Data_Type_Stats 뷰의 원본 쿼리를 활용하면, 질환(우울증, 파킨슨병)별로 어떤 데이터 유형이 몇 건씩 수집되었는지 한눈에 파악할 수 있습니다.

각 질환별 데이터 통계 쿼리
SQL

SELECT
    data_category,   -- 질환 구분 (예: 우울증, 파킨슨병)
    data_type,       -- 데이터 유형 (예: 설문지, 영상, 생체 데이터)
    COUNT(*) AS item_count  -- 해당 유형의 총 건수
FROM
    tb_Clinical_Data_Items
GROUP BY
    data_category,
    data_type
ORDER BY
    data_category,
    data_type;

CREATE TABLE tb_Questionnaire_Questions (
    question_id INT AUTO_INCREMENT PRIMARY KEY,          -- 질문의 고유 식별자
    questionnaire_type VARCHAR(50),                      -- 설문지 유형 (예: 우울증-A형, 우울증-B형)
    question_number INT,                                 -- 설문지 내 질문 번호
    question_text VARCHAR(500)                           -- 질문 내용
);

CREATE TABLE tb_Questionnaire_Answers (
    answer_id INT AUTO_INCREMENT PRIMARY KEY,          -- 답변의 고유 식별자
    item_id INT,                                       -- 외래 키: 어떤 설문지 항목에 대한 답변인지 연결
    question_id INT,                                   -- 외래 키: 어떤 질문에 대한 답변인지 연결
    answer_component VARCHAR(50),                      -- 답변의 세부 항목 (예: '왼쪽다리', '오른쪽팔')
    answer_value VARCHAR(255),                         -- 답변 내용 (점수, 텍스트 등)
    submission_datetime DATETIME,                      -- 답변 제출 시간
    FOREIGN KEY (item_id) REFERENCES tb_Clinical_Data_Items(item_id),
    FOREIGN KEY (question_id) REFERENCES tb_Questionnaire_Questions(question_id)
);


// ------------ 파킨슨병 중증도

CREATE TABLE tb_Parkinson_Stage (
    stage_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    stage_value DECIMAL(2, 1),                           -- 0, 1, 1.5, 2, 2.5, ...
    stage_description TEXT,                              -- 질병의 증후가 없음, 일측성 상하지 장애 등
    assessment_date DATE,                                -- 단계가 평가된 날짜
    FOREIGN KEY (patient_id) REFERENCES tb_Patient_Info(patient_id)
);



// -------------- 다운로드 이력관리
다운로드 이력 관리 테이블 (제안)
테이블명: tb_Download_History

설명: 어떤 데이터가, 누가, 언제 다운로드되었는지에 대한 정보를 기록합니다.

CREATE TABLE tb_Download_History (
    download_id INT AUTO_INCREMENT PRIMARY KEY, -- 다운로드 이벤트의 고유 식별자
    data_type VARCHAR(50),                      -- 다운로드된 데이터의 유형 (예: '학습 데이터', '샘플 데이터')
    download_user_id VARCHAR(50),               -- 다운로드를 수행한 사용자의 ID 또는 식별자
    download_datetime DATETIME                  -- 다운로드가 발생한 시점
);



// ----------------플랫폼 관리

1. 시스템 모니터링 및 서비스 상태 관리 테이블
시스템의 현재 상태와 발생한 이벤트를 기록하는 테이블입니다. 서비스의 정상 작동 여부, 오류 발생 등을 실시간으로 모니터링하는 데 활용할 수 있습니다.

테이블명: tb_System_Status_Log

설명: 시스템의 주요 서비스 상태 변화 및 로그를 기록합니다.

CREATE TABLE tb_System_Status_Log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,          -- 로그 고유 식별자
    service_name VARCHAR(100) NOT NULL,              -- 모니터링 대상 서비스 이름 (예: '데이터 수집 API', '전처리 엔진')
    status_level VARCHAR(20) NOT NULL,               -- 상태 수준 (예: '정상', '경고', '오류')
    event_message TEXT,                              -- 발생한 이벤트에 대한 상세 메시지
    event_timestamp DATETIME NOT NULL                -- 이벤트 발생 시각
);



2. 스케줄 관리 테이블
정기적으로 실행되는 백업, 데이터 처리, 통계 생성 등 자동화된 작업들을 관리하는 테이블입니다.

테이블명: tb_Job_Schedule
CREATE TABLE tb_Job_Schedule (
    job_id INT AUTO_INCREMENT PRIMARY KEY,         -- 작업 고유 식별자
    job_name VARCHAR(100) NOT NULL,                -- 작업 이름 (예: '일일 데이터 백업', '주간 통계 생성')
    schedule_expression VARCHAR(255),              -- 스케줄 주기 (예: '매일 새벽 2시', '매주 일요일')
    last_run_timestamp DATETIME,                   -- 마지막으로 실행된 시각
    next_run_timestamp DATETIME,                   -- 다음 실행 예정 시각
    status VARCHAR(20) NOT NULL,                   -- 현재 상태 (예: '대기중', '실행중', '성공', '실패')
    last_run_log TEXT                              -- 마지막 실행 결과 로그
);




3. 저장소 현황 관리 테이블
시스템이 사용하는 물리적 또는 클라우드 저장소의 현재 상태를 기록하는 테이블입니다.

테이블명: tb_Storage_Status

설명: 저장소의 전체 용량, 사용량, 남은 용량 등을 주기적으로 기록합니다.

CREATE TABLE tb_Storage_Status (
    status_id INT AUTO_INCREMENT PRIMARY KEY,          -- 상태 기록 고유 식별자
    storage_name VARCHAR(100) NOT NULL,                -- 저장소 이름 (예: '데이터-스토리지-01', '영상-S3-버킷')
    total_space_gb DECIMAL(10, 2) NOT NULL,            -- 전체 용량 (GB 단위)
    used_space_gb DECIMAL(10, 2) NOT NULL,             -- 사용된 용량 (GB 단위)
    used_percentage DECIMAL(5, 2) NOT NULL,            -- 사용률 (%)
    last_updated_at DATETIME NOT NULL                  -- 마지막 업데이트 시각
);

2. 백업/복구 이력 관리 테이블
데이터 백업 및 복구 작업의 성공/실패 여부를 기록하고, 관련 정보를 추적하는 테이블입니다. 이는 tb_Job_Schedule 테이블의 상세 로그 역할을 할 수도 있습니다.

테이블명: tb_Backup_History

설명: 백업 및 복구 작업의 실행 이력을 기록합니다

CREATE TABLE tb_Backup_History (
    backup_id INT AUTO_INCREMENT PRIMARY KEY,          -- 백업 작업 고유 식별자
    backup_type VARCHAR(50) NOT NULL,                  -- 작업 유형 (예: '전체 백업', '증분 백업', '복구')
    status VARCHAR(20) NOT NULL,                       -- 작업 상태 (예: '성공', '실패', '진행 중')
    start_time DATETIME NOT NULL,                      -- 작업 시작 시각
    end_time DATETIME,                                 -- 작업 종료 시각
    backup_location VARCHAR(255),                      -- 백업 파일 저장 위치
    backup_size_gb DECIMAL(10, 2),                     -- 백업된 데이터 크기 (GB 단위)
    error_message TEXT                                 -- 실패 시 오류 메시지
);

3. 임계치 설정 테이블
임계치처럼 자주 변경되지 않으면서 시스템 동작의 기준이 되는 값은 별도의 테이블에 저장하는 것이 데이터베이스 정규화 원칙에 부합하며 관리 효율성을 극대화합니다.

테이블명: tb_Storage_Config

설명: 저장소 관련 설정 값을 한 번만 저장합니다.
CREATE TABLE tb_Storage_Config (
    config_id INT AUTO_INCREMENT PRIMARY KEY,        -- 설정 항목 고유 식별자
    storage_name VARCHAR(100) UNIQUE NOT NULL,       -- 대상 저장소 이름 (예: '데이터-스토리지-01')
    alert_threshold_gb DECIMAL(10, 2),               -- 경고 임계치 (GB 단위)
    alert_threshold_percentage DECIMAL(5, 2)         -- 경고 임계치 (%)
);


4. 저장소 경고 관리 테이블
저장소 사용량이 임계치를 초과하는 등 중요한 경고 이벤트가 발생했을 때 기록하는 테이블입니다. 이는 tb_System_Status_Log와 유사하나, 저장소에 특화된 정보를 담습니다.

테이블명: tb_Storage_Alerts

설명: 저장소 관련 경고 이벤트를 기록합니다.
CREATE TABLE tb_Storage_Alerts (
    alert_id INT AUTO_INCREMENT PRIMARY KEY,           -- 경고 고유 식별자
    storage_name VARCHAR(100) NOT NULL,                -- 경고가 발생한 저장소 이름
    alert_type VARCHAR(50) NOT NULL,                   -- 경고 유형 (예: '용량 부족', '읽기 실패')
    threshold_value DECIMAL(5, 2),                     -- 경고 임계치 값
    current_value DECIMAL(10, 2),                      -- 현재 값 (예: 사용률 %)
    alert_message TEXT,                                -- 경고 상세 메시지
    alert_timestamp DATETIME NOT NULL                  -- 경고 발생 시각
);

