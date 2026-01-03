# EarthML

## Usage

## SSL / HTTPS access (HPC & enterprise environments)

earthML does not modify SSL or certificate settings at runtime.

On some HPC or enterprise systems, HTTPS endpoints are signed by CAs that are not included in the default system trust store. In these cases, Python libraries such as requests or earthkit may fail with:

```
SSL: CERTIFICATE_VERIFY_FAILED
```

Create a custom CA bundle that includes the missing issuer (e.g. HARICA / GEANT) and point Python to it:

```bash
export REQUESTS_CA_BUNDLE=$HOME/certs/earthml-ca-bundle.pem
export SSL_CERT_FILE=$HOME/certs/earthml-ca-bundle.pem
```

Then run your experiment using the project virtual environment, e.g.:

```bash
.venv/bin/python launch_seasonal_ocean_exp.py
```

Example: building a custom CA bundle
```bash
cat /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
    $HOME/certs/harica_tls_rsa_root_ca_2021.pem \
    > $HOME/certs/earthml-ca-bundle.pem
```

(Ask your system administrator if you are unsure which CA certificate is required.)

Why this is required

Certificate trust is environment-specific on HPC systems.
For safety and portability, earthml leaves SSL configuration to the user or job environment.

### Sanity check

```bash
REQUESTS_CA_BUNDLE=$HOME/certs/earthml-ca-bundle.pem \
SSL_CERT_FILE=$HOME/certs/earthml-ca-bundle.pem \
.venv/bin/python - <<'EOF'
import requests
requests.get("https://object-store.os-api.cci2.ecmwf.int", timeout=10)
print("SSL OK")
EOF
```

### Obtaining the required CA certificate (ECMWF / CCI2)

On ECMWF / CCI2 systems, the object store is currently signed by the
**HARICA / GEANT** certificate chain.

If the required root certificate is not present in your system trust store,
you can extract it directly from the server using OpenSSL:

```bash
openssl s_client -showcerts \
  -connect object-store.os-api.cci2.ecmwf.int:443 \
  -servername object-store.os-api.cci2.ecmwf.int </dev/null
```
