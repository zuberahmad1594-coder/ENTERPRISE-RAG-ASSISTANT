"""
Generates a small, coherent, self-authored HR document collection for the
Enterprise Document Intelligence & RAG Assistant capstone project.

These are synthetic (fictional-company) HR documents created specifically for
this project so there are no copyright / confidentiality concerns. Replace
the contents of data/documents/ with your own organization's public or
permissioned documents if you prefer.
"""

import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=12)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceAfter=8, spaceBefore=14)
body = ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=8, leading=15)


def build_pdf(filename, title, sections):
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=LETTER,
                             topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                             leftMargin=0.9 * inch, rightMargin=0.9 * inch)
    story = [Paragraph(title, h1), Spacer(1, 12)]
    for heading, paragraphs in sections:
        story.append(Paragraph(heading, h2))
        for p in paragraphs:
            story.append(Paragraph(p, body))
    doc.build(story)
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# 1. Employee Handbook
# ---------------------------------------------------------------------------
build_pdf(
    "employee_handbook.pdf",
    "NorthBridge Solutions — Employee Handbook",
    [
        ("1. Introduction", [
            "This Employee Handbook applies to all full-time and part-time employees of "
            "NorthBridge Solutions Pvt. Ltd. It summarizes the company's key policies, "
            "expectations, and benefits. This handbook supplements, and does not replace, "
            "the employment agreement signed by each employee.",
        ]),
        ("2. Employment Classification", [
            "Employees are classified as Full-Time, Part-Time, Contract, or Intern. "
            "Full-time employees work a minimum of 40 hours per week and are eligible for "
            "the full benefits package described in Section 6. Interns are eligible for a "
            "limited subset of benefits as described in their offer letter.",
        ]),
        ("3. Working Hours", [
            "Standard working hours are 9:30 AM to 6:30 PM, Monday through Friday, with a "
            "one-hour lunch break. Core collaboration hours, during which all employees are "
            "expected to be reachable, are 11:00 AM to 4:00 PM local time.",
        ]),
        ("4. Resignation and Notice Period", [
            "Employees who wish to resign must submit written notice to their manager and "
            "to Human Resources. The standard notice period is 60 days for employees at the "
            "Associate level and above, and 30 days for employees below the Associate level. "
            "The company may, at its discretion, offer a buyout of the notice period.",
        ]),
        ("5. Probation Period", [
            "New employees serve a probation period of 90 days from their date of joining. "
            "During probation, either the employee or the company may terminate employment "
            "with 15 days' written notice. Confirmation of employment occurs automatically at "
            "the end of the probation period unless the employee is notified otherwise.",
        ]),
        ("6. Benefits Overview", [
            "Full-time employees are eligible for group health insurance covering the "
            "employee, spouse, and up to two dependent children. The company also provides "
            "term life insurance equal to two times annual base salary, and an annual health "
            "check-up reimbursement of up to INR 5,000.",
            "Detailed benefit terms, including waiting periods and claim procedures, are "
            "described in the separate Benefits Documentation.",
        ]),
        ("7. Code of Conduct Reference", [
            "All employees are required to comply with the company's Code of Conduct, "
            "which is provided as a separate document and covers workplace behavior, "
            "conflicts of interest, confidentiality, and anti-harassment policy.",
        ]),
        ("8. Grievance Redressal", [
            "Employees with workplace grievances should first raise the issue with their "
            "direct manager. If unresolved within 10 working days, the grievance may be "
            "escalated to Human Resources via the internal HR portal.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 2. Leave Policy
# ---------------------------------------------------------------------------
build_pdf(
    "leave_policy.pdf",
    "NorthBridge Solutions — Leave Policy",
    [
        ("1. Purpose", [
            "This policy defines the types of leave available to employees of NorthBridge "
            "Solutions, the process for requesting leave, and the accrual and carry-forward "
            "rules that apply to each leave type.",
        ]),
        ("2. Annual (Earned) Leave", [
            "Full-time employees are entitled to 18 days of Annual Leave per calendar year, "
            "accrued at a rate of 1.5 days per completed month of service. Annual Leave must "
            "be requested at least 3 working days in advance through the HR system, except in "
            "emergencies.",
        ]),
        ("3. Sick Leave", [
            "Employees are entitled to 10 days of Sick Leave per calendar year. Sick Leave "
            "exceeding 2 consecutive days requires a medical certificate submitted within 5 "
            "working days of returning to work.",
        ]),
        ("4. Casual Leave", [
            "Employees may take up to 6 days of Casual Leave per calendar year for short-notice "
            "personal matters. Casual Leave cannot be combined with Annual Leave for a period "
            "exceeding 5 consecutive working days without manager approval.",
        ]),
        ("5. Maternity and Paternity Leave", [
            "Eligible employees are entitled to 26 weeks of paid Maternity Leave in accordance "
            "with applicable law. Paternity Leave of 10 working days is available to eligible "
            "employees within 6 months of the child's birth or adoption.",
        ]),
        ("6. Carry Forward and Encashment", [
            "Up to 10 unused Annual Leave days may be carried forward to the following calendar "
            "year. Unused Sick Leave and Casual Leave do not carry forward and lapse at year-end. "
            "Employees may encash a maximum of 5 carried-forward Annual Leave days per year, "
            "subject to manager and HR approval.",
        ]),
        ("7. Unpaid Leave", [
            "Employees who have exhausted their applicable leave balance may request Unpaid "
            "Leave, subject to manager and HR approval. Unpaid Leave exceeding 30 consecutive "
            "days requires Director-level approval and may affect benefit continuity.",
        ]),
        ("8. Public Holidays", [
            "NorthBridge Solutions observes 12 public holidays per calendar year, as published "
            "annually on the HR portal. Employees required to work on a public holiday are "
            "entitled to a compensatory day off within 30 days.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 3. Work-From-Home Policy
# ---------------------------------------------------------------------------
build_pdf(
    "work_from_home_policy.pdf",
    "NorthBridge Solutions — Work-From-Home (WFH) Policy",
    [
        ("1. Policy Statement", [
            "NorthBridge Solutions supports a hybrid working model that balances business "
            "needs with employee flexibility. This policy defines eligibility, scheduling, "
            "equipment, and expectations for employees working from home.",
        ]),
        ("2. Eligibility", [
            "All full-time employees who have completed their probation period are eligible "
            "for the hybrid work arrangement, subject to role requirements. Roles that require "
            "physical presence (e.g., front-desk, warehouse operations) are excluded and "
            "governed by their department-specific policy.",
        ]),
        ("3. Standard Hybrid Schedule", [
            "The default hybrid schedule requires employees to work from the office a minimum "
            "of 3 days per week, with the remaining days available for remote work. Managers "
            "may, with HR approval, authorize a fully remote arrangement for specific roles.",
        ]),
        ("4. Requesting Fully Remote Work", [
            "Employees seeking a fully remote arrangement (5 days per week) must submit a "
            "request through the HR portal along with manager endorsement. Requests are "
            "reviewed on a case-by-case basis considering role, performance, and team needs.",
        ]),
        ("5. Equipment and Reimbursement", [
            "The company provides a laptop and standard peripherals for remote work. Employees "
            "working from home are eligible for an internet reimbursement of up to INR 1,500 "
            "per month upon submission of a valid bill.",
        ]),
        ("6. Availability and Communication", [
            "Employees working from home are expected to be reachable during core collaboration "
            "hours (11:00 AM–4:00 PM) via company-approved communication tools and to attend all "
            "scheduled meetings with camera on unless otherwise agreed with their manager.",
        ]),
        ("7. Data Security While Working Remotely", [
            "Employees must use company-issued devices and an approved VPN when accessing "
            "internal systems remotely. Working from public Wi-Fi networks without VPN "
            "protection is prohibited. Any suspected security incident must be reported to IT "
            "Security within 24 hours.",
        ]),
        ("8. Policy Review", [
            "This policy is reviewed annually by Human Resources and may be updated to reflect "
            "changes in business needs or applicable regulation. Employees will be notified of "
            "material changes at least 30 days before they take effect.",
        ]),
    ],
)

# ---------------------------------------------------------------------------
# 4. Code of Conduct
# ---------------------------------------------------------------------------
build_pdf(
    "code_of_conduct.pdf",
    "NorthBridge Solutions — Code of Conduct",
    [
        ("1. Purpose and Scope", [
            "This Code of Conduct sets out the standards of behavior expected of every "
            "NorthBridge Solutions employee, contractor, and intern, regardless of role or "
            "location. Violations may result in disciplinary action up to and including "
            "termination of employment.",
        ]),
        ("2. Professional Conduct", [
            "Employees are expected to treat colleagues, clients, and partners with respect "
            "and professionalism. Discrimination or harassment based on race, gender, religion, "
            "age, disability, sexual orientation, or any other protected characteristic will "
            "not be tolerated.",
        ]),
        ("3. Anti-Harassment Policy", [
            "NorthBridge Solutions maintains a zero-tolerance policy toward workplace "
            "harassment, including sexual harassment. Employees who experience or witness "
            "harassment should report it immediately to HR or via the confidential ethics "
            "hotline. All reports are investigated promptly and confidentially.",
        ]),
        ("4. Conflicts of Interest", [
            "Employees must disclose any outside business activity, financial interest, or "
            "personal relationship that could reasonably be seen to conflict with the "
            "interests of NorthBridge Solutions. Disclosures should be submitted to HR via "
            "the Conflict of Interest form.",
        ]),
        ("5. Confidentiality of Information", [
            "Employees must protect confidential company, client, and employee information "
            "both during and after their employment. Confidential information must not be "
            "shared with unauthorized parties or used for personal gain.",
        ]),
        ("6. Use of Company Assets", [
            "Company equipment, systems, and resources are provided for business purposes. "
            "Reasonable personal use is permitted provided it does not interfere with work "
            "duties or violate any other company policy.",
        ]),
        ("7. Anti-Bribery and Gifts", [
            "Employees must not offer, give, solicit, or accept bribes or improper payments. "
            "Business gifts exceeding INR 2,000 in value must be declared to the Compliance "
            "team and may require declination or donation.",
        ]),
        ("8. Reporting Violations", [
            "Employees who become aware of a violation of this Code of Conduct should report "
            "it through their manager, HR, or the anonymous ethics hotline. Retaliation "
            "against anyone who reports a concern in good faith is strictly prohibited.",
        ]),
    ],
)

print("\nAll sample HR documents generated.")
