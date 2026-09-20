# Target roles (WHOOP Spring 2027 co-op, Jan 11 - Jul 1, 2027)

Main target, role 1: **Performance Science & Health Outcomes Co-Op**. Fallback, role 2: **Data Analytics Co-Op**.
Source: the two job postings on jobs.ashbyhq.com/whoop (re-read them before applying; wording can change).

## Role 1 requirements -> where the portfolio shows it
| Posting asks for | Evidence in this repo |
|---|---|
| Research on real-world physiological data | P3 core analysis on public wearable data; P1 real CMJ data |
| SQL queries to retrieve datasets | P2 SQL layer (.sql files, tested against pandas) |
| t-tests, ANOVA, regression; plus mixed models, causal inference, mediation | P1 variance components, P2 mixed model, P3 mixed models + causal-thinking section |
| Python (R is a plus) | Python throughout; mixed models in R (lme4) if preferred |
| Git, reproducible workflows, data hygiene | Pinned dependencies, `make all`, DATA_SOURCES.md, de-identification, commit history showing the plan before the analysis |
| Formulate hypotheses, draw conclusions, present clearly | P3 written analysis plan, 2-page research brief, plain-language summaries |
| Literature review / wearable and performance science | Reference list in each README; short literature note in the brief |
| Experience with human participants | The user's 10-athlete CMJ dataset (state clearly what role the user had in collecting it) |
| Using AI tools with the same quality standard | README section on how AI was used and verified |

## Role 2 (fallback) requirements -> extra
Product/marketing/growth analytics: metrics, A/B tests and incrementality, dashboards, SQL/Python/BI, business
communication. The Streamlit dashboard, SQL layer and statistics already help. Add the stretch packs S4 (A/B test
analysis on a simulated experiment) and S5 (metric definitions + engagement-style dashboard) if applying here.

## Things the user must verify herself (the agent cannot)
- Work authorisation and location: the postings mention E-Verify; confirm eligibility, location and remote status
  with the recruiter before investing more.
- Enrolment status and dates: confirm she qualifies as a co-op student for Jan 11 - Jul 1, 2027 and how her college
  internship window fits (the postings invite raising schedule constraints).
- Permission to use and publish the CMJ data.
