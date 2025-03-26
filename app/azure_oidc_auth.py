import os
import jwt
import requests
import time  # Importing time module to get the system's Unix time
from datetime import datetime, timezone
from jwt.exceptions import InvalidTokenError

from models import AuthTokenRequest
from urllib.parse import urlencode
from fastapi.responses import JSONResponse
from fastapi_server_session import Session

AZ_CLIENT_ID = os.getenv("AZ_CLIENT_ID")
AZ_CLIENT_SECRET = os.getenv("AZ_CLIENT_SECRET")
AZ_TENANT_ID = os.getenv("AZ_TENANT_ID")

def signup(request:AuthTokenRequest, session: Session):
    claims = validate_id_token(request.id_token, os.getenv("AZ_JWKS_URI"), AZ_CLIENT_ID)
    print(claims)
    if claims.get("sub") and claims.get("name"):
        session["user_id"] = claims["sub"]
        session["user_name"] = claims["name"]
    data = {
        "grant_type": "authorization_code",
        "code": request.code,
        "client_id": AZ_CLIENT_ID,
        "client_secret": AZ_CLIENT_SECRET,
        "redirect_uri": "http://localhost:3000",
        "scope": "openid profile offline_access https://graph.microsoft.com/.default"
    }
    encoded = urlencode(data)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(f"https://login.microsoftonline.com/{AZ_TENANT_ID}/oauth2/v2.0/token", headers=headers, data=encoded)
    res_body = response.json()
    session["access_token"] = res_body["access_token"]
    session["refresh_token"] = res_body["refresh_token"]
    session["id_token"] = res_body["id_token"]
    session["expires_at"] = int(time.time()) + res_body["expires_in"]
    return {"id_token": session["id_token"], "expires": session["expires_at"]}

def check_session(session: Session):
    if "expires_at" in session and session["expires_at"] - int(time.time()) < 60:
        return JSONResponse(
            status_code=403,
            content={"error": "Session expired"}
        )


def refresh_session(session: Session, id_token: str):
    if "id_token" in session and session["id_token"] != id_token:
        print("ID token mismatch, possible spoofing attempt!")
        return JSONResponse(status_code=403,content={"result": "error", "message": "ID token mismatch. Who are you?"})
    try:
        claims = validate_id_token(id_token, os.getenv("AZ_JWKS_URI"), AZ_CLIENT_ID)
        if claims.get("sub") != session["user_id"]:
            print("ID token subject claim mismatch, possible spoofing attempt!")
            return JSONResponse(status_code=403,content={"result": "error", "message": "Subject mismatch. Who are you?"})
    except Exception as e:
        print(f"ID token validation failed: {str(e)}")
        return JSONResponse(status_code=403,content={"result": "error", "message": "ID token validation failed"})
    if "expires_at" in session and session["expires_at"] - int(time.time()) < 60:
        data = {
            "grant_type": "refresh_token",
            "client_id": AZ_CLIENT_ID,
            "client_secret": AZ_CLIENT_SECRET,
            "refresh_token": session["refresh_token"],
            "scope": "openid profile offline_access https://graph.microsoft.com/.default"
        }
        encoded = urlencode(data)
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = requests.post(f"https://login.microsoftonline.com/{AZ_TENANT_ID}/oauth2/v2.0/token", headers=headers, data=encoded)
        res_body = response.json()
        session["access_token"] = res_body["access_token"]
        session["refresh_token"] = res_body["refresh_token"]
        session["id_token"] = res_body["id_token"]
        session["expires_at"] = int(time.time()) + res_body["expires_in"]
        return {"result": "success"}
    return JSONResponse(status=200,content={"result": "success"})

def validate_id_token(id_token, jwks_uri, client_id):
    """
    Extract and validate claims from an OIDC ID token.
    
    Args:
        id_token: The JWT token string
        jwks_uri: URI to the JWKS (JSON Web Key Set) endpoint
        client_id: Your application's client ID
        
    Returns:
        dict: The validated token claims if valid
        
    Raises:
        Exception: If token validation fails
    """
    # Get the public keys from the JWKS endpoint
    jwks_response = requests.get(jwks_uri)
    jwks = jwks_response.json()
    
    # Decode the token header (without verification)
    # to get the key ID (kid) used to sign the token
    token_header = jwt.get_unverified_header(id_token)
    kid = token_header.get('kid')
    
    # Find the matching key from JWKS
    public_key = None
    for key in jwks['keys']:
        if key['kid'] == kid:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            break
    
    if not public_key:
        raise Exception("Public key not found")
    
    try:
        # Decode and verify the token
        claims = jwt.decode(
            id_token,
            public_key,
            algorithms=['RS256'],
            options={
                'verify_signature': True,
                'verify_exp': True,
                'verify_iat': True,
                'verify_aud': True,
                'require': ['exp', 'iat', 'iss', 'aud']
            },
            audience=client_id
        )
        
        # Additional validation checks
        current_time = datetime.now(timezone.utc).timestamp()
        
        # Check issuer (you may want to validate against your trusted issuer)
        # claims['iss'] should match your expected issuer
        
        # Check if token is not yet valid (optional)
        if 'nbf' in claims and claims['nbf'] > current_time:
            raise Exception("Token not yet valid")
        
        # Check subject is present
        if 'sub' not in claims:
            raise Exception("Subject claim missing")
            
        return claims
        
    except InvalidTokenError as e:
        raise Exception(f"Token validation failed: {str(e)}")