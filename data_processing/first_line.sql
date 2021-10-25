with egfr as (
    select
        m.person_id,
        m.value_as_number as val,
        row_number() over (partition by m.person_id order by m.measurement_datetime desc) as rk
    from
        cdm.measurement m
    join
        experiments.t2dm_matchcohort mc
    on
        m.person_id = mc.person_id
    where
        m.measurement_concept_id in (3053283, 3049187, 46236952, 40764999)
    and
        mc.index_date >= m.measurement_datetime
    and
        DATE_PART('day', mc.index_date-m.measurement_datetime) <= 180
),

heart_disease as (
    select distinct
        co.person_id,
        1 as occurred
    from
        cdm.condition_occurrence co
    join
        experiments.t2dm_matchcohort mc
    on
        co.person_id = mc.person_id
    where
        co.condition_concept_id in (
            select
                c.concept_id
            from cdm.concept c
            join cdm.concept_ancestor ca
            on c.concept_id = ca.descendant_concept_id
            where ca.ancestor_concept_id = 316139  -- heart failure
        )
    and
        mc.index_date >= co.condition_start_datetime
    and
        DATE_PART('day', mc.index_date-co.condition_start_datetime) <= 730
),

creatinine as (
    select
        m.person_id,
        m.value_as_number as val,
        row_number() over (partition by m.person_id order by m.measurement_datetime desc) as rk
    from
        cdm.measurement m
    join
        experiments.t2dm_matchcohort mc
    on
        m.person_id = mc.person_id
    where
        m.measurement_concept_id in (3016723)
    and
        mc.index_date >= m.measurement_datetime
    and
        DATE_PART('day', mc.index_date-m.measurement_datetime) <= 180
)
    

select
    mc.person_id,
    egfr.val as egfr,
    coalesce(hd.occurred, 0) as heart_disease,
    creatinine.val as creatinine,
    mc.index_date as treatment_date,
    case 
        when lower(mc.ingredient_name) like '%metformin%' then 0
        when (lower(mc.ingredient_name) like '%glipizide%' or
              lower(mc.ingredient_name) like '%glimepiride%' or
              lower(mc.ingredient_name) like '%sitagliptin%' or
              lower(mc.ingredient_name) like '%glyburide%') then 1
    end as y
from experiments.t2dm_matchcohort mc
left join egfr
on
    mc.person_id = egfr.person_id
left join heart_disease hd
on 
    mc.person_id = hd.person_id
left join creatinine
on
    mc.person_id = creatinine.person_id
where 
    egfr.rk = 1
and
    creatinine.rk = 1
and 
    (lower(mc.ingredient_name) like '%glipizide%' or
     lower(mc.ingredient_name) like '%glimepiride%' or
     lower(mc.ingredient_name) like '%sitagliptin%' or
     lower(mc.ingredient_name) like '%glyburide%' or
     lower(mc.ingredient_name) like '%metformin%')
;