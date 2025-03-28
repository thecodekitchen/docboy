import os
import jwt
import requests
import time  # Importing time module to get the system's Unix time
import dotenv

from datetime import datetime, timezone
from jwt.exceptions import InvalidTokenError

from urllib.parse import urlencode
from fastapi.responses import JSONResponse
from fastapi_server_session import Session
from pydantic import BaseModel
from logging import Logger
class AuthTokenRequest(BaseModel):
    id_token: str
    code: str
    session_state: str

class RefreshRequest(BaseModel):
    id_token: str

class OauthOIDCConfig(BaseModel):
    client_id: str
    client_secret: str
    jwks_uri: str
    auth_uri: str
    token_uri: str
    redirect_uri: str
    discovery_uri: str | None = None
    userinfo_uri: str | None = None
    logout_uri: str | None = None
    scope: str = "openid profile offline_access email"

dotenv.load_dotenv()
oauth_config = OauthOIDCConfig(
    client_id=os.getenv("KC_CLIENT_ID"),
    client_secret=os.getenv("KC_CLIENT_SECRET"),
    jwks_uri=os.getenv("KC_JWKS_URI"),
    auth_uri=os.getenv("KC_AUTH_URI"),
    token_uri=os.getenv("KC_TOKEN_URI"),
    redirect_uri=os.getenv("KC_REDIRECT_URI"),
    userinfo_uri=os.getenv("KC_USERINFO_URI"),
    logout_uri=os.getenv("KC_LOGOUT_URI"),
    discovery_uri=os.getenv("KC_DISCOVERY_URI")
)

def login(request:AuthTokenRequest, session: Session, logger: Logger):
    claims = validate_id_token(request.id_token, oauth_config.jwks_uri, oauth_config.client_id)
    logger.debug("validated id claims: "+ str(claims))
    if claims.get("sub") and claims.get("name"):
        session["user_id"] = claims["sub"]
        session["user_name"] = claims["name"]
    data = {
        "grant_type": "authorization_code",
        "code": request.code,
        "client_id": oauth_config.client_id,
        "client_secret": oauth_config.client_secret,
        "redirect_uri": "http://localhost:3000/",
        "scope": oauth_config.scope
    }
    encoded = urlencode(data)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(oauth_config.token_uri, headers=headers, data=encoded)
    res_body = response.json()
    logger.debug("access token response: " + str(res_body))
    if res_body.get("access_token") and res_body.get("refresh_token") and res_body.get("id_token"):
        session["access_token"] = res_body["access_token"]
        session["refresh_token"] = res_body["refresh_token"]
        session["id_token"] = res_body["id_token"]
        new_id_claims = validate_id_token(session["id_token"], oauth_config.jwks_uri, oauth_config.client_id)
        if new_id_claims.get("exp"):
            logger.debug("refreshed id claims: "+ str(new_id_claims))
            session["id_expires"] = int(time.time())+new_id_claims["exp"]
            session["expires_at"] = int(time.time()) + res_body["expires_in"]
            res = JSONResponse(status_code=200,content={"id_token": session["id_token"], "expires": session["expires_at"]})
            res.set_cookie("session", session.session_id)
            logger.debug("login response: "+ str(res.__dict__))
            return res
        else:
            logger.error("ID token missing expiry")
            return JSONResponse(status_code=403,content={"result": "error", "message": "ID token missing expiry"})
    else:
        logger.error("Access token failed")
        return JSONResponse(status_code=403,content={"result": "error", "message": "Access token failed: "+str(res_body)})


def check_session(session: Session, logger: Logger):
    if "expires_at" in session and session["expires_at"] - int(time.time()) < 60:
        logger.error("Session expired")
        return JSONResponse(
            status_code=403,
            content={"error": "Session expired"}
        )


def refresh_session(session: Session, id_token: str, logger: Logger):
    if "id_token" in session and session["id_token"] != id_token:
        logger.error("ID token mismatch, possible spoofing attempt!")
        return JSONResponse(status_code=403,content={"result": "error", "message": "ID token mismatch. Who are you?"})
    try:
        claims = validate_id_token(id_token, oauth_config.jwks_uri, oauth_config.client_id)
        if claims.get("sub") != session["user_id"]:
            logger.error("ID token subject claim mismatch, possible spoofing attempt!")
            return JSONResponse(status_code=403,content={"result": "error", "message": "Subject mismatch. Who are you?"})
    except Exception as e:
        logger.error(f"ID token validation failed: {str(e)}")
        return JSONResponse(status_code=403,content={"result": "error", "message": "ID token validation failed"})
    if "expires_at" in session and session["expires_at"] - int(time.time()) < 60:
        data = {
            "grant_type": "refresh_token",
            "client_id": oauth_config.client_id,
            "client_secret": oauth_config.client_secret,
            "refresh_token": session["refresh_token"],
            "scope": oauth_config.scope
        }
        encoded = urlencode(data)
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = requests.post(oauth_config.token_uri, headers=headers, data=encoded)
        res_body = response.json()
        logger.debug("refresh token response: ", res_body)
        if res_body.get("access_token") and res_body.get("refresh_token") and res_body.get("id_token"):
            session["access_token"] = res_body["access_token"]
            session["refresh_token"] = res_body["refresh_token"]
            session["id_token"] = res_body["id_token"]
            new_id_claims = validate_id_token(session["id_token"], oauth_config.jwks_uri, oauth_config.client_id)
            logger.debug("refreshed id claims: ", new_id_claims)
            if new_id_claims.get("exp"):
                session["id_expires"] = int(time.time())+new_id_claims["exp"]
                session["expires_at"] = int(time.time()) + res_body["expires_in"]
                res = JSONResponse(status_code=200,content={"id_token": session["id_token"], "expires": session["expires_at"]})
                res.set_cookie("session", session.session_id)
                return res
            else:
                logger.error("ID token missing expiry")
                return JSONResponse(status_code=403,content={"result": "error", "message": "ID token missing expiry"})
        else:
            logger.error("Refresh token failed")
            return JSONResponse(status_code=403,content={"result": "error", "message": "Refresh token failed: "+str(res_body)})
    res = JSONResponse(status_code=200,content={"id_token": session["id_token"], "expires": session["expires_at"]})
    res.set_cookie("session", session.session_id)
    return res

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