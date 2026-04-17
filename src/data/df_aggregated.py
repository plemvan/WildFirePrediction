import os
import pandas as pd
import s3fs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_data_from_s3() -> pd.DataFrame:
    """
    Loads the aggregated wildfire dataset directly from S3 (MinIO).
    Requires S3_ENDPOINT_URL and S3_BUCKET to be set in the environment.
    """
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': os.getenv('S3_ENDPOINT_URL')}
    )
    
    bucket = os.getenv('S3_BUCKET')
    file_path = f"{bucket}/df_aggregated.parquet"
    
    with fs.open(file_path, 'rb') as f:
        return pd.read_csv(f)
