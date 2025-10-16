-- dev_kkh.tb_questionnaire_answers definition

-- Drop table

-- DROP TABLE dev_kkh.tb_questionnaire_answers;
-- dev_kkh.tb_questionnaire_questions definition

-- Drop table

-- DROP TABLE dev_kkh.tb_questionnaire_questions;

CREATE TABLE dev_kkh.tb_questionnaire_questions ( question_id serial4 NOT NULL, questionnaire_type varchar(50) NULL, question_number int4 NULL, question_text varchar(500) NULL, CONSTRAINT tb_questionnaire_questions_pkey PRIMARY KEY (question_id));
COMMENT ON TABLE dev_kkh.tb_questionnaire_questions IS '설문지 질문 마스터 테이블';

-- Column comments

COMMENT ON COLUMN dev_kkh.tb_questionnaire_questions.question_id IS '질문 ID';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_questions.questionnaire_type IS '설문지 유형';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_questions.question_number IS '질문 번호';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_questions.question_text IS '질문 내용';

-- Permissions

ALTER TABLE dev_kkh.tb_questionnaire_questions OWNER TO kkh;
GRANT INSERT, TRUNCATE, TRIGGER, REFERENCES, SELECT, DELETE, UPDATE ON TABLE dev_kkh.tb_questionnaire_questions TO kkh;
GRANT INSERT, SELECT, DELETE, UPDATE ON TABLE dev_kkh.tb_questionnaire_questions TO pd_dep_collector;
GRANT INSERT, SELECT, DELETE, UPDATE ON TABLE dev_kkh.tb_questionnaire_questions TO agi_cms;



CREATE TABLE dev_kkh.tb_questionnaire_answers ( answer_id serial4 NOT NULL, item_id int4 NOT NULL, question_id int4 NOT NULL, answer_component varchar(50) NULL, answer_value varchar(255) NULL, submission_datetime timestamp DEFAULT '2025-10-10 08:28:30.715048'::timestamp without time zone NULL, CONSTRAINT tb_questionnaire_answers_pkey PRIMARY KEY (answer_id));
COMMENT ON TABLE dev_kkh.tb_questionnaire_answers IS '설문지 질문 테이블';

-- Column comments

COMMENT ON COLUMN dev_kkh.tb_questionnaire_answers.answer_id IS '답변 ID';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_answers.item_id IS '외래 키 : 데이터 항목 식별자';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_answers.question_id IS '외래 키 : 질문 ID';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_answers.answer_component IS '답변 세부 항목';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_answers.answer_value IS '답변 내용';
COMMENT ON COLUMN dev_kkh.tb_questionnaire_answers.submission_datetime IS '답변 제출 시각';

-- Permissions

ALTER TABLE dev_kkh.tb_questionnaire_answers OWNER TO kkh;
GRANT INSERT, TRUNCATE, TRIGGER, REFERENCES, SELECT, DELETE, UPDATE ON TABLE dev_kkh.tb_questionnaire_answers TO kkh;
GRANT INSERT, SELECT, DELETE, UPDATE ON TABLE dev_kkh.tb_questionnaire_answers TO pd_dep_collector;
GRANT INSERT, SELECT, DELETE, UPDATE ON TABLE dev_kkh.tb_questionnaire_answers TO agi_cms;


-- dev_kkh.tb_questionnaire_answers foreign keys

ALTER TABLE dev_kkh.tb_questionnaire_answers ADD CONSTRAINT tb_questionnaire_answers_item_id_fkey FOREIGN KEY (item_id) REFERENCES dev_kkh.tb_items(item_id);
ALTER TABLE dev_kkh.tb_questionnaire_answers ADD CONSTRAINT tb_questionnaire_answers_question_id_fkey FOREIGN KEY (question_id) REFERENCES dev_kkh.tb_questionnaire_questions(question_id);