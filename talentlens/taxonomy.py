"""Shared public knowledge: domains, skills, aliases, scoring helpers used by BOTH the
ground-truth oracle and the graders, so the rubric is identical everywhere."""
from __future__ import annotations

DOMAINS = {
    "data_science": dict(
        titles=["Data Scientist", "Machine Learning Engineer", "Applied Scientist", "Data Analyst", "Research Scientist"],
        skills=["Python", "SQL", "PyTorch", "scikit-learn", "Statistics", "Experiment Design", "Spark", "NLP", "Deep Learning", "Pandas"],
        fields=["Computer Science", "Statistics", "Mathematics", "Physics"],
        certs=["AWS ML Specialty", "Databricks ML Associate", "TensorFlow Developer"],
        tasks=["churn prediction models", "recommendation systems", "demand forecasting pipelines", "fraud detection models", "search ranking experiments"]),
    "backend": dict(
        titles=["Backend Engineer", "Software Engineer", "Platform Engineer", "Site Reliability Engineer", "API Developer"],
        skills=["Python", "Go", "Java", "PostgreSQL", "Kubernetes", "Docker", "REST APIs", "Kafka", "AWS", "System Design"],
        fields=["Computer Science", "Software Engineering", "Electrical Engineering"],
        certs=["AWS Solutions Architect", "CKA", "Terraform Associate"],
        tasks=["payment services", "event streaming platforms", "multi-region deployments", "internal developer platforms", "high-traffic public APIs"]),
    "frontend": dict(
        titles=["Frontend Engineer", "UI Engineer", "Web Developer", "Mobile Engineer", "Full Stack Engineer"],
        skills=["JavaScript", "TypeScript", "React", "CSS", "Accessibility", "Next.js", "GraphQL", "Figma", "Web Performance", "Testing Library"],
        fields=["Computer Science", "Design", "Software Engineering"],
        certs=["Meta Front-End Certificate", "Google UX Certificate", "WAS Accessibility"],
        tasks=["design systems", "checkout flows", "dashboards", "marketing sites", "progressive web apps"]),
    "finance": dict(
        titles=["Financial Analyst", "Investment Analyst", "FP&A Manager", "Equity Research Associate", "Risk Analyst"],
        skills=["Financial Modeling", "Excel", "Valuation", "DCF", "Bloomberg", "Forecasting", "Risk Management", "SQL", "Accounting", "M&A"],
        fields=["Finance", "Economics", "Accounting", "Mathematics"],
        certs=["CFA", "FRM", "CPA"],
        tasks=["valuation models", "quarterly forecasts", "acquisition due diligence", "portfolio risk reports", "budgeting processes"]),
    "legal": dict(
        titles=["Corporate Counsel", "Legal Associate", "Compliance Officer", "Contract Manager", "Privacy Counsel"],
        skills=["Contract Drafting", "GDPR", "Due Diligence", "Regulatory Compliance", "Legal Research", "IP Law", "Negotiation", "Litigation Support", "Privacy Law", "Corporate Governance"],
        fields=["Law", "Political Science", "Business"],
        certs=["CIPP/E", "Bar Admission", "CIPM"],
        tasks=["vendor contracts", "data protection programmes", "M&A diligence workstreams", "compliance audits", "licensing agreements"]),
    "clinical": dict(
        titles=["Clinical Research Associate", "Biostatistician", "Medical Writer", "Clinical Data Manager", "Pharmacovigilance Specialist"],
        skills=["GCP", "Clinical Trial Design", "SAS", "R", "Medical Writing", "Regulatory Submissions", "EDC Systems", "Biostatistics", "Pharmacovigilance", "CDISC"],
        fields=["Biology", "Medicine", "Pharmacy", "Statistics"],
        certs=["ACRP-CP", "SOCRA CCRP", "RAC"],
        tasks=["phase II oncology trials", "safety reporting pipelines", "submission dossiers", "statistical analysis plans", "site monitoring programmes"]),
    "marketing": dict(
        titles=["Growth Marketer", "Marketing Manager", "SEO Specialist", "Content Strategist", "Performance Marketer"],
        skills=["SEO", "Google Analytics", "Paid Acquisition", "Copywriting", "A/B Testing", "CRM", "Email Marketing", "Brand Strategy", "SQL", "Attribution Modeling"],
        fields=["Marketing", "Business", "Communications", "Economics"],
        certs=["Google Ads Certification", "HubSpot Inbound", "Meta Blueprint"],
        tasks=["lifecycle campaigns", "paid social programmes", "organic growth strategies", "conversion funnels", "brand relaunches"]),
}

# Aliases BOTH the grader and the generator know about. (The generator also injects typos,
# which the rule grader cannot resolve -- a realistic gap an LLM grader should close.)
ALIASES = {
    "Python": ["python3", "python programming"], "SQL": ["t-sql", "sql queries"],
    "scikit-learn": ["sklearn"], "PyTorch": ["torch"], "Deep Learning": ["neural networks"],
    "Statistics": ["statistical modeling"], "Spark": ["pyspark", "apache spark"],
    "Experiment Design": ["experimentation"], "Kubernetes": ["k8s"], "REST APIs": ["restful services"],
    "JavaScript": ["js", "ecmascript"], "TypeScript": ["ts"], "React": ["react.js", "reactjs"],
    "Financial Modeling": ["financial models", "3-statement modeling"], "Excel": ["microsoft excel"],
    "GDPR": ["eu data protection"], "SEO": ["search engine optimization"],
    "A/B Testing": ["ab testing", "split testing"], "Google Analytics": ["ga4"],
    "Clinical Trial Design": ["trial design"], "Valuation": ["company valuation"],
    "Paid Acquisition": ["paid media"], "Web Performance": ["core web vitals"],
}

LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]
DEGREE_RANK = {"BSc": 1, "BA": 1, "LLB": 1, "MSc": 2, "MA": 2, "LLM": 2, "MBA": 2, "PhD": 3, "JD": 3}
EU_COUNTRIES = {"Germany", "Italy", "Spain", "Poland", "France", "Netherlands", "Ireland"}


def score_years(observed: float, required: float) -> int:
    return 2 if observed >= required else 1 if observed >= 0.6 * required else 0


def score_degree(rank: int, min_rank: int) -> int:
    return 2 if rank >= min_rank else 1 if rank == min_rank - 1 else 0


def score_language(level: str, minimum: str) -> int:
    a, b = LEVEL_ORDER.index(level), LEVEL_ORDER.index(minimum)
    return 2 if a >= b else 1 if a == b - 1 else 0
