const API_URL = "http://127.0.0.1:8000";


function base64urlToUint8Array(base64url: string): Uint8Array {
  const padding = "=".repeat(
    (4 - (base64url.length % 4)) % 4
  );

  const base64 = (base64url + padding)
    .replace(/-/g, "+")
    .replace(/_/g, "/");

  const binaryString = window.atob(base64);

  const bytes = new Uint8Array(
    binaryString.length
  );

  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return bytes;
}


function arrayBufferToBase64url(
  buffer: ArrayBuffer
): string {
  const bytes = new Uint8Array(buffer);

  let binaryString = "";

  for (const byte of bytes) {
    binaryString += String.fromCharCode(byte);
  }

  return window
    .btoa(binaryString)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}


export async function registerPasskey(): Promise<{
  success: boolean;
  message: string;
}> {
  if (!window.PublicKeyCredential) {
    return {
      success: false,
      message:
        "WebAuthn is not supported by this browser.",
    };
  }


  try {
    // ========================================================
    // STEP 1 — REQUEST REGISTRATION OPTIONS
    // ========================================================

    const optionsResponse = await fetch(
      `${API_URL}/auth/register/options`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );


    if (!optionsResponse.ok) {
      throw new Error(
        `Registration options request failed with status ${optionsResponse.status}.`
      );
    }


    const options = await optionsResponse.json();


    // ========================================================
    // STEP 2 — CONVERT SERVER DATA FOR WEBAUTHN
    // ========================================================

    const publicKey:
      PublicKeyCredentialCreationOptions = {
      ...options,

      challenge: base64urlToUint8Array(
        options.challenge
      ),

      user: {
        ...options.user,

        id: base64urlToUint8Array(
          options.user.id
        ),
      },

      excludeCredentials: (
        options.excludeCredentials ?? []
      ).map(
        (credential: {
          id: string;
          type: PublicKeyCredentialType;
        }) => ({
          ...credential,

          id: base64urlToUint8Array(
            credential.id
          ),
        })
      ),
    };


    // ========================================================
    // STEP 3 — ASK THE BROWSER / DEVICE TO CREATE PASSKEY
    // ========================================================

    const credential =
      (await navigator.credentials.create({
        publicKey,
      })) as PublicKeyCredential | null;


    if (!credential) {
      return {
        success: false,
        message:
          "Passkey registration was cancelled.",
      };
    }


    // ========================================================
    // STEP 4 — SERIALIZE CREDENTIAL
    // ========================================================

    const response =
      credential.response as AuthenticatorAttestationResponse;


    const credentialData = {
      id: credential.id,

      rawId: arrayBufferToBase64url(
        credential.rawId
      ),

      type: credential.type,

      response: {
        clientDataJSON:
          arrayBufferToBase64url(
            response.clientDataJSON
          ),

        attestationObject:
          arrayBufferToBase64url(
            response.attestationObject
          ),
      },
    };


    // ========================================================
    // STEP 5 — SEND CREDENTIAL TO BACKEND
    // ========================================================

    const verificationResponse = await fetch(
      `${API_URL}/auth/register/verify`,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify(
          credentialData
        ),
      }
    );


    if (!verificationResponse.ok) {
      throw new Error(
        `Passkey verification request failed with status ${verificationResponse.status}.`
      );
    }


    const verification =
      await verificationResponse.json();


    // ========================================================
    // STEP 6 — RETURN RESULT
    // ========================================================

    if (!verification.success) {
      return {
        success: false,
        message:
          verification.message ||
          "Passkey verification failed.",
      };
    }


    return {
      success: true,

      message:
        "Passkey registered and verified successfully.",
    };


  } catch (error) {
    console.error(
      "WebAuthn registration error:",
      error
    );


    return {
      success: false,

      message:
        error instanceof Error
          ? error.message
          : "Passkey registration failed.",
    };
  }
}
export async function authenticateWithPasskey(): Promise<{
  success: boolean;
  message: string;
}> {
  if (!window.PublicKeyCredential) {
    return {
      success: false,
      message: "WebAuthn is not supported by this browser.",
    };
  }

  try {
    // --------------------------------------------------------
    // GET AUTHENTICATION OPTIONS
    // --------------------------------------------------------

    const optionsResponse = await fetch(
      `${API_URL}/auth/login/options`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (!optionsResponse.ok) {
      throw new Error(
        `Authentication options request failed with status ${optionsResponse.status}.`
      );
    }

    const options = await optionsResponse.json();

    if (options.success === false) {
      return {
        success: false,
        message:
          options.message ||
          "No registered passkey was found.",
      };
    }

    // --------------------------------------------------------
    // CONVERT SERVER OPTIONS TO BROWSER FORMAT
    // --------------------------------------------------------

    const publicKey:
      PublicKeyCredentialRequestOptions = {
      ...options,
      challenge: base64urlToUint8Array(
        options.challenge
      ),
      allowCredentials: (
        options.allowCredentials ?? []
      ).map(
        (credential: {
          id: string;
          type: PublicKeyCredentialType;
        }) => ({
          ...credential,
          id: base64urlToUint8Array(
            credential.id
          ),
        })
      ),
    };

    // --------------------------------------------------------
    // ASK AUTHENTICATOR FOR PASSKEY
    // --------------------------------------------------------

    const credential =
      (await navigator.credentials.get({
        publicKey,
      })) as PublicKeyCredential | null;

    if (!credential) {
      return {
        success: false,
        message: "Passkey authentication was cancelled.",
      };
    }

    // --------------------------------------------------------
    // EXTRACT AUTHENTICATION RESPONSE
    // --------------------------------------------------------

    const response =
      credential.response as AuthenticatorAssertionResponse;

    const credentialData = {
      id: credential.id,

      rawId: arrayBufferToBase64url(
        credential.rawId
      ),

      type: credential.type,

      response: {
        clientDataJSON:
          arrayBufferToBase64url(
            response.clientDataJSON
          ),

        authenticatorData:
          arrayBufferToBase64url(
            response.authenticatorData
          ),

        signature:
          arrayBufferToBase64url(
            response.signature
          ),

        userHandle:
          response.userHandle
            ? arrayBufferToBase64url(
                response.userHandle
              )
            : null,
      },
    };

    // --------------------------------------------------------
    // SEND AUTHENTICATION RESPONSE TO BACKEND
    // --------------------------------------------------------

    const verificationResponse = await fetch(
      `${API_URL}/auth/login/verify`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          credentialData
        ),
      }
    );

    if (!verificationResponse.ok) {
      throw new Error(
        `Passkey verification request failed with status ${verificationResponse.status}.`
      );
    }

    const verification =
      await verificationResponse.json();

    if (!verification.success) {
      return {
        success: false,
        message:
          verification.message ||
          "Passkey authentication failed.",
      };
    }

    return {
      success: true,
      message:
        "Passkey authentication successful.",
    };

  } catch (error) {

    console.error(
      "WebAuthn authentication error:",
      error
    );

    return {
      success: false,
      message:
        error instanceof Error
          ? error.message
          : "Passkey authentication failed.",
    };
  }
}