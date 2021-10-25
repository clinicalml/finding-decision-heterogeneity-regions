/*

Cohort query. Modified diabetes cohort definition from OHDSI treatment pathways study. Intended to define a cohort with first-line diabetes decisions:

1. No need for post-index observation period (i.e. after first treatment, we don't care how long they were observed)
2. Require at least one A1c measurement >= 6.5 within [1y pre-index, 1y post-index]
3. Exclusion conditions are expanded to additional codes based on concept names.

Summary of cohort inclusion criteria:
1. Required 3-year observation before first diabetes treatment.
2. At least one diagnosis code of "Diabetes mellitus" during that observation.
3. No diagnosis code related to type 1 diabetes mellitus during that observation or gestational diabetes in 1 year prior to treatment.
4. At least one A1c measurement of >= 6.5 during that observation.
*/

SET search_path TO experiments;

DROP TABLE IF EXISTS T2DM_indexcohort;
DROP TABLE IF EXISTS T2DM_t0;
DROP TABLE IF EXISTS T2DM_t1;
DROP TABLE IF EXISTS T2DM_m;
DROP TABLE IF EXISTS T2DM_matchcohort;

CREATE TEMP TABLE T2DM_IndexCohort
 (PERSON_ID bigint not null primary key,
    INDEX_DATE date not null,
    INGREDIENT_NAME varchar not null,
    PROVIDER_ID bigint not null,
    OBSERVATION_PERIOD_START_DATE date not null,
    OBSERVATION_PERIOD_END_DATE date not null
);

INSERT INTO T2DM_IndexCohort (PERSON_ID, INDEX_DATE, INGREDIENT_NAME, PROVIDER_ID, OBSERVATION_PERIOD_START_DATE, OBSERVATION_PERIOD_END_DATE)

-- For each patient, take end date to be minimum end date where every drug exposure that started prior to that has ended by that date.

select  person_id, INDEX_DATE, INGREDIENT_NAME, PROVIDER_ID, observation_period_start_date, observation_period_end_date
FROM 
(
    select ot.PERSON_ID, ot.INDEX_DATE, ot.INGREDIENT_NAME, ot.PROVIDER_ID, ot.OBSERVATION_PERIOD_START_DATE, ot.OBSERVATION_PERIOD_END_DATE, ROW_NUMBER() OVER (PARTITION BY ot.PERSON_ID ORDER BY ot.INDEX_DATE) as RowNumber
    
    -- Find the first T2DM drug exposure and set it to the index date. Require observation period containing that drug exposure to be at least 1y before index date, and 1y after.
    
    from 
    (
        select dt.PERSON_ID, dt.DRUG_EXPOSURE_START_DATE as index_date, c.concept_name as INGREDIENT_NAME, dt.PROVIDER_ID, op.OBSERVATION_PERIOD_START_DATE, op.OBSERVATION_PERIOD_END_DATE
        from  
        (
            select de.PERSON_ID, de.DRUG_CONCEPT_ID, de.DRUG_EXPOSURE_START_DATE, de.PROVIDER_ID
            FROM 
            (
                select d.PERSON_ID, d.DRUG_CONCEPT_ID, d.DRUG_EXPOSURE_START_DATE, d.PROVIDER_ID,
                COALESCE(d.DRUG_EXPOSURE_END_DATE, (d.DRUG_EXPOSURE_START_DATE + d.DAYS_SUPPLY*INTERVAL'1 day'), (d.DRUG_EXPOSURE_START_DATE + 1*INTERVAL'1 day')) as DRUG_EXPOSURE_END_DATE,
                ROW_NUMBER() OVER (PARTITION BY d.PERSON_ID ORDER BY DRUG_EXPOSURE_START_DATE) as RowNumber
                FROM cdm.DRUG_EXPOSURE d
                JOIN cdm.CONCEPT_ANCESTOR ca 
                on d.DRUG_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID and ca.ANCESTOR_CONCEPT_ID in (21600712)
            ) de
            JOIN cdm.PERSON p on p.PERSON_ID = de.PERSON_ID
            WHERE de.RowNumber = 1
        ) dt
        JOIN cdm.observation_period op 
            on op.PERSON_ID = dt.PERSON_ID and (dt.DRUG_EXPOSURE_START_DATE between op.OBSERVATION_PERIOD_START_DATE and op.OBSERVATION_PERIOD_END_DATE)
        JOIN cdm.drug_strength ds
            on ds.drug_concept_id = dt.drug_concept_id
        JOIN cdm.concept c
            on c.concept_id = ds.ingredient_concept_id
        
        -- 3 years observation prior.
        WHERE (op.OBSERVATION_PERIOD_START_DATE + 1095*INTERVAL'1 day') <= dt.DRUG_EXPOSURE_START_DATE AND (dt.DRUG_EXPOSURE_START_DATE) <= op.OBSERVATION_PERIOD_END_DATE

    ) ot
    GROUP BY ot.PERSON_ID, ot.INDEX_DATE, ot.INGREDIENT_NAME, ot.PROVIDER_ID, ot.OBSERVATION_PERIOD_START_DATE, ot.OBSERVATION_PERIOD_END_DATE
) r
WHERE r.RowNumber = 1
;

--find persons in indexcohort with diagnosis
CREATE TEMP TABLE T2DM_T1
 (PERSON_ID bigint not null primary key,
    INDEX_DATE date not null
);

INSERT INTO T2DM_T1
select ip.PERSON_ID, ip.INDEX_DATE
from T2DM_IndexCohort ip
LEFT JOIN 
(
    select ce.PERSON_ID, ce.CONDITION_CONCEPT_ID
    FROM derived_tables.CONDITION_ERA ce
    JOIN T2DM_IndexCohort ip on ce.PERSON_ID = ip.PERSON_ID
    JOIN cdm.CONCEPT_ANCESTOR ca on ce.CONDITION_CONCEPT_ID = ca.DESCENDANT_CONCEPT_ID and ca.ANCESTOR_CONCEPT_ID in (201820)
    WHERE (ce.CONDITION_ERA_START_DATETIME between ip.OBSERVATION_PERIOD_START_DATE and ip.OBSERVATION_PERIOD_END_DATE)
) ct on ct.PERSON_ID = ip.PERSON_ID
GROUP BY  ip.PERSON_ID, ip.INDEX_DATE
HAVING COUNT(ct.CONDITION_CONCEPT_ID) >= 1
;

-- find persons in indexcohort with >=1 A1c measurement >= 6.5
CREATE TEMP TABLE T2DM_M
 (PERSON_ID bigint not null primary key,
    INDEX_DATE date not null
);

INSERT INTO T2DM_M
select ip.PERSON_ID, ip.INDEX_DATE
from T2DM_IndexCohort ip
LEFT JOIN
(
    select m.PERSON_ID, m.MEASUREMENT_CONCEPT_ID
    FROM cdm.MEASUREMENT m
    JOIN T2DM_IndexCohort ip on m.PERSON_ID = ip.PERSON_ID
    WHERE (m.MEASUREMENT_DATE between ip.OBSERVATION_PERIOD_START_DATE and ip.OBSERVATION_PERIOD_END_DATE
        AND m.MEASUREMENT_CONCEPT_ID in (3004410, 3005673, 3034639)
        AND m.VALUE_AS_NUMBER >= 6.5)
) dt on dt.PERSON_ID = ip.PERSON_ID
GROUP BY  ip.PERSON_ID, ip.INDEX_DATE
HAVING COUNT(dt.MEASUREMENT_CONCEPT_ID) >= 1
;

--find persons that qualify meet all inclusion criteria
create table T2DM_MatchCohort_Without_Exclusion
(
    PERSON_ID bigint not null primary key,
    INDEX_DATE date not null,
    INGREDIENT_NAME varchar not null,
    PROVIDER_ID bigint not null,
    OBSERVATION_PERIOD_START_DATE date not null,
    OBSERVATION_PERIOD_END_DATE date not null
);


INSERT INTO T2DM_MatchCohort_Without_Exclusion (PERSON_ID, INDEX_DATE, INGREDIENT_NAME, PROVIDER_ID, OBSERVATION_PERIOD_START_DATE, OBSERVATION_PERIOD_END_DATE)
select c.person_id, c.index_date, c.ingredient_name, c.provider_id, c.observation_period_start_date, c.observation_period_end_date
FROM T2DM_IndexCohort C
INNER JOIN
(
SELECT INDEX_DATE, PERSON_ID
FROM
(
    SELECT INDEX_DATE, PERSON_ID FROM T2DM_T1
    INTERSECT
    SELECT INDEX_DATE, PERSON_ID FROM T2DM_M
) TopGroup
) I 
ON C.PERSON_ID = I.PERSON_ID
and c.index_date = i.INDEX_DATE
;

-- Exclude patients with T1DM prior to index date
CREATE TABLE T1DM_To_Exclude AS
SELECT DISTINCT p.person_id
FROM T2DM_MatchCohort_Without_Exclusion p
JOIN cdm.condition_occurrence co
ON co.person_id = p.person_id
AND co.condition_start_datetime <= TO_DATE(p.index_date,'YYYY-MM-DD')
JOIN cdm.concept c
ON co.condition_concept_id = c.concept_id
AND (LOWER(c.concept_name) LIKE '%type 1 diabet%'
     OR LOWER(c.concept_name) LIKE '%type i diabet%'
     OR LOWER(c.concept_name) LIKE '%diabet%type 1%'
     OR LOWER(c.concept_name) LIKE '%diabet%type i %')
AND LOWER(c.concept_name) NOT LIKE '%type 2 diabet%'
AND LOWER(c.concept_name) NOT LIKE '%type ii diabet%'
AND LOWER(c.concept_name) NOT LIKE '%diabet%type 2%'
AND LOWER(c.concept_name) NOT LIKE '%diabet%type ii%'
AND LOWER(c.concept_name) NOT LIKE '%gestat%'
AND LOWER(c.concept_name) NOT LIKE '%pregnan%'
AND LOWER(c.concept_name) NOT LIKE '%diabetes of the young%'
AND LOWER(c.concept_name) NOT LIKE '%neonatal%diabet%'
AND LOWER(c.concept_name) NOT LIKE '%diabet%neonatal%';
        
-- Exclude patients with gestational diabetes within 1 year prior to index date
CREATE TABLE Gestational_Diabetes_To_Exclude AS
SELECT DISTINCT p.person_id
FROM T2DM_MatchCohort_Without_Exclusion p
JOIN cdm.condition_occurrence co
ON co.person_id = p.person_id
AND co.condition_start_datetime <= TO_DATE(p.index_date,'YYYY-MM-DD')
AND co.condition_start_datetime > TO_DATE(p.index_date,'YYYY-MM-DD') - interval '1 year'
JOIN cdm.concept c
ON co.condition_concept_id = c.concept_id
AND (LOWER(c.concept_name) LIKE '%pregnan%'
     OR LOWER(c.concept_name) LIKE '%gestat%'
     OR LOWER(c.concept_name) LIKE '%diabetes of the young%'
     OR LOWER(c.concept_name) LIKE '%neonatal%diabet%'
     OR LOWER(c.concept_name) LIKE '%diabet%neonatal%');
        
-- Create final cohort with exclusion
CREATE TABLE T2DM_MatchCohort AS
SELECT DISTINCT p.person_id, p.index_date
FROM T2DM_MatchCohort_Without_Exclusion p
WHERE NOT EXISTS (
    SELECT t1dm.person_id
    FROM T1DM_To_Exclude t1dm
    WHERE t1dm.person_id = p.person_id
) AND NOT EXISTS (
    SELECT pregnancy.person_id
    FROM Gestational_Diabetes_To_Exclude pregnancy
    WHERE p.person_id = pregnancy.person_id
);