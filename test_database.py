from sqlmodel import Session, select
from api.database import engine
from api.models import Prediction


# Open a database session to retrieve stored prediction records.
with Session(engine) as session:

    # Build a query to retrieve all predictions, ordered by ID from newest to oldest.
    statement = select(Prediction).order_by(Prediction.id.desc())

    # Execute the query and retrieve all matching prediction records.
    predictions = session.exec(statement).all()

    # Display the stored information for each prediction.
    for prediction in predictions:
        print(f"Prediction ID: {prediction.id}")
        print(f"Original image: {prediction.original_image_path}")
        print(f"Annotated image: {prediction.annotated_image_path}")
        print(f"Status: {prediction.status}")
        print("-" * 40)
