from config import engine, SessionLocal
from models.db_models import Base, Newspaper, Section

# List of newspapers to add
newspapers = [
    {"name": "Página 12", "url": "https://www.pagina12.com.ar/"},
    {"name": "TN", "url": "https://tn.com.ar/"},
    {"name": "Perfil", "url": "https://www.perfil.com/"},
    {"name": "Infobae", "url": "https://www.infobae.com/"},
    {"name": "El Economista", "url": "https://eleconomista.com.ar/"},
    {"name": "Ámbito Financiero", "url": "https://www.ambito.com/"},
]

sections = [
    'Economía',
    'Internacional',
    'Política',
    'Sociedad',
]


def insert_newspapers():
    # Start a new database session
    db_session = SessionLocal()

    for paper in newspapers:
        # Check if the newspaper already exists
        existing_newspaper = db_session.query(Newspaper).filter_by(name=paper["name"]).first()
        if not existing_newspaper:
            # Create and add a new Newspaper object if it doesn't exist
            new_newspaper = Newspaper(name=paper["name"], url=paper["url"])
            db_session.add(new_newspaper)
            print(f"Inserted newspaper: {paper['name']}")
        else:
            print(f"Newspaper already exists: {paper['name']}")

    # Commit the transaction to the database
    db_session.commit()
    db_session.close()


def insert_sections():
    # Start a new database session
    db_session = SessionLocal()

    for section in sections:
        # Check if the section already exists
        existing_section = db_session.query(Section).filter_by(name=section).first()
        if not existing_section:
            # Create and add a new Section object if it doesn't exist
            new_section = Section(name=section)
            db_session.add(new_section)
            print(f"Inserted newspaper: {section}")
        else:
            print(f"Section already exists: {section}")

    # Commit the transaction to the database
    db_session.commit()
    db_session.close()


if __name__ == "__main__":
    # Create the tables in the database
    print("Creating tables...")
    Base.metadata.create_all(engine)
    print("Tables created successfully.")

    # Insert the predefined newspapers
    print("Inserting newspapers...")
    insert_newspapers()
    print("Newspapers inserted successfully.")

    # Insert the predefined sections
    print("Inserting sections...")
    insert_sections()
    print("Sections inserted successfully.")
