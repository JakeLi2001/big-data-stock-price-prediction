# import libraries
import zipfile
import boto3
from io import BytesIO

bucket="cis4130-project-jakeli"
zipfile_to_unzip="x"
s3_client = boto3.client('s3', use_ssl=False)
s3_resource = boto3.resource('s3')

zip_obj = s3_resource.Object(bucket_name=bucket, key=zipfile_to_unzip)
buffer = BytesIO(zip_obj.get()["Body"].read())
z = zipfile.ZipFile(buffer)
# Loop through all of the files contained in the Zip archive
for filename in z.namelist():
    print('Working on ' + filename)
    # Unzip the file and write it back to S3 in the same bucket
    s3_resource.meta.client.upload_fileobj(z.open(filename),Bucket=bucket,Key=f'{filename}')