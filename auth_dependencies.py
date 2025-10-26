
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import Client, create_client
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Supabase client (make sure env vars are set)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# This defines how FastAPI expects the token (in the Authorization header)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl isn't used directly here

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency to verify the JWT token and return the user object.
    Raises HTTPException 401 if the token is invalid or missing.
    """
    try:
        # Supabase automatically removes "Bearer " if present
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except Exception as e:
        print(f"Auth Error: {e}") # Log the specific error
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# You can add more dependencies later, e.g., to check roles/tiers
# async def get_admin_user(current_user: User = Depends(get_current_user)):
#    # Query profiles table, check if current_user.id has 'admin' tier
#    # If not, raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not an admin")
#    return current_user
