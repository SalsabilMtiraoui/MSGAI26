
from haystack import Document

documents = [
    Document(
        content="""
        The Swiss Joint Master of Science in Computer Science is a
        joint study programme offered by the universities of Bern,
        Neuchatel and Fribourg.

        The programme offers more than 60 courses and seminars.
        Students can create a personalized curriculum and specialize
        in areas of computer science.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/",
            "section": "Programme overview",
        },
    ),
    Document(
        content="""
        The standard Master's programme consists of 90 ECTS credits.

        It comprises 60 ECTS credits from teaching units and
        30 ECTS credits from the Master's thesis.

        Students must complete 12 teaching units, each worth
        5 ECTS credits.

        The programme normally takes 3 semesters of full-time study.
        Part-time students can request an extension.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/program/",
            "section": "Programme structure and duration",
        },
    ),
    Document(
        content="""
        The programme offers 7 tracks reflecting different profiles
        in computer science.

        Students must complete teaching units from at least
        3 different tracks.

        Students wishing to specialize in one track must complete
        at least 5 teaching units from that track and write their
        Master's thesis on a related topic.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/program/",
            "section": "Tracks and specialization",
        },
    ),
    Document(
        content="""
        The official teaching language of the Joint Master's
        programme is English.

        Courses are usually taught in English, but some courses
        may be taught in French or German depending on the professor.

        Students who only speak English can follow the programme
        without problems.

        The programme does not require language tests or
        language certificates.

        An English proficiency level of at least B2 (CEFR)
        is strongly recommended.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/program/",
            "section": "Teaching languages",
        },
    ),
    Document(
        content="""
        The Joint Master's programme is open to students holding
        a Bachelor of Science in Computer Science from a Swiss
        university or an equivalent degree from a foreign university.

        Students with a Computer Science degree from a University
        of Applied Sciences may also apply. In principle, they
        should have a grade average of at least 5.0.

        Students with other types of university degrees, particularly
        degrees with a minor in Computer Science, may also apply.

        Applications in these cases are evaluated individually.
        The admissions committee may require additional undergraduate
        courses, up to a maximum of 60 ECTS credits.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/admission/",
            "section": "Admission requirements",
        },
    ),
    Document(
        content="""
        Students can choose any of the three participating universities
        as their home university: Bern, Neuchatel or Fribourg.

        The home university is responsible for administrative matters,
        including registration, payment of fees and diploma ceremonies.

        Regardless of the selected home university, academic admission
        decisions are handled by the same branch committee.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/admission/",
            "section": "Choosing a home university",
        },
    ),
    Document(
        content="""
        Students can start the Joint Master's programme in either
        the autumn or spring semester.

        For autumn admission, the application deadline is April 30
        for applicants who do not require a visa, and February 28
        for applicants who require a visa at the University of Fribourg.

        For spring admission, the deadline is November 30 at
        Neuchatel and Fribourg, and December 15 at Bern for
        applicants who do not require a visa.

        For visa applicants, the spring deadline is September 30
        at the University of Fribourg.

        Applicants should check the relevant registration office
        for current requirements.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/admission/",
            "section": "Application deadlines",
        },
    ),
    Document(
        content="""
        The programme offers more than 60 teaching units, consisting
        of courses and seminars.

        Students choose 12 teaching units.

        The tracks organize courses according to different areas
        of computer science.

        The programme is designed to allow students to build a
        personalized curriculum according to their interests.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/program/tracks/",
            "section": "Teaching units and tracks",
        },
    ),
    Document(
        content="""
        The programme timetable is divided into 10 half-day
        time slots distributed across the five weekdays.

        Students can take at most two courses per day.

        The timetable is organized to minimize conflicts between
        teaching units belonging to the same track.

        Courses are distributed across the autumn and spring semesters.
        Seminars may have more flexible schedules and are not
        necessarily included in the general timetable.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/program/courses-timetable/",
            "section": "Timetable",
        },
    ),
    Document(
        content="""
        There are no summer courses in the Joint Master's programme.

        Courses are offered during the autumn and spring semesters.

        Students can take language courses at their home university.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/faq/",
            "section": "Courses and language courses",
        },
    ),
    Document(
        content="""
        Students can study part-time.

        The programme does not formally distinguish between
        full-time and part-time students.

        The maximum study duration is normally limited to
        six semesters.

        Students can request an extension from the Branch Committee
        if they have a justified reason.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/faq/",
            "section": "Part-time studies",
        },
    ),
    Document(
        content="""
        Travel expenses between the participating universities
        are reimbursed by the student's home university.

        The programme involves universities in Bern, Neuchatel
        and Fribourg.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/faq/",
            "section": "Travel reimbursement",
        },
    ),
    Document(
        content="""
        Students can continue to a PhD after completing the
        Joint Master's degree.

        Students interested in doctoral studies should consult
        the websites of the different research groups.
        """,
        meta={
            "source": "https://mcs.unibnf.ch/faq/",
            "section": "Doctoral studies",
        },
    ),
]
