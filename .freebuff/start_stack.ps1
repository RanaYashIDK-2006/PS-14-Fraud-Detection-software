$py = 'C:\Users\Rana Yash Singh\OneDrive\Documents\Project\.venv\Scripts\python.exe'
$wd = 'C:\Users\Rana Yash Singh\OneDrive\Documents\Project'

# --- Load .env file into environment ---
$envFile = Join-Path $wd '.env'
if (Test-Path $envFile) {
  Get-Content $envFile | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' } | ForEach-Object {
    $parts = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
  }
  Write-Output "Loaded .env ($envFile)"
}

# --- Local stack only: never run in Postgres mode -----------------------
# .env may carry DATABASE_URL / SUPABASE_* (Supabase remote setup). This
# launcher is the LOCAL SQLite preview - settings.use_postgres is driven by
# DATABASE_URL, so strip the Postgres vars from the child environment or
# every service tries Supabase and dies (DNS/credentials). Postgres mode is
# docker-compose.prod.yml's job.
Remove-Item Env:DATABASE_URL, Env:SUPABASE_URL, Env:SUPABASE_ANON_KEY, Env:SUPABASE_SERVICE_KEY, Env:DB_SCHEMA -ErrorAction SilentlyContinue

# --- Detect LAN IP ---
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -ne '127.0.0.1' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress
if (-not $lanIp) { $lanIp = '127.0.0.1' }
Write-Output "LAN IP: $lanIp"

# --- CORS origins: HTTP only (no self-signed certs) ---
$origins = @(
  "http://127.0.0.1:8000", "http://127.0.0.1:8004",
  "http://localhost:8000", "http://localhost:8004",
  "http://${lanIp}:8000", "http://${lanIp}:8004"
) -join ','
[Environment]::SetEnvironmentVariable('CORS_ORIGINS', $origins, 'Process')

# --- Set service URLs (HTTP) ---
[Environment]::SetEnvironmentVariable('IDENTITY_URL', "http://${lanIp}:8001", 'Process')
[Environment]::SetEnvironmentVariable('PYTHONPATH', "$wdackend", 'Process')
[Environment]::SetEnvironmentVariable('PRIVACY_URL',  "http://${lanIp}:8002", 'Process')
[Environment]::SetEnvironmentVariable('RISK_URL',      "http://${lanIp}:8003", 'Process')
[Environment]::SetEnvironmentVariable('AUDIT_URL',     "http://${lanIp}:8005", 'Process')
[Environment]::SetEnvironmentVariable('VERIFY_URL',    "http://${lanIp}:8004", 'Process')

$services = @(
  @{name='front';    port=8000; mod='src.front_service.main:app'},
  @{name='identity'; port=8001; mod='src.identity_service.main:app'},
  @{name='privacy';  port=8002; mod='src.privacy_layer.main:app'},
  @{name='risk';     port=8003; mod='src.risk_engine.main:app'},
  @{name='verify';   port=8004; mod='src.verification_service.main:app'},
  @{name='audit';    port=8005; mod='src.audit_service.main:app'}
)

# Idempotent: stop existing listeners
foreach ($s in $services) {
  Get-NetTCPConnection -LocalPort $s.port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue } |
    Where-Object { $_.ProcessName -match 'python|uvicorn' } |
    ForEach-Object { Write-Output "stopping $($s.name) (pid $($_.Id))"; Stop-Process -Id $_.Id -Force }
}
Start-Sleep -Milliseconds 800

# Start via Python subprocess (HTTP, no SSL)
$launchScript = @"
import subprocess, sys, os, time
py = r'$py'
wd = r'$wd'
services = [
$(foreach ($s in $services) { "    ('$($s.name)', $($s.port), '$($s.mod)')," })
]
for name, port, mod in services:
    env = dict(os.environ); env['PYTHONPATH'] = os.path.join(wd, 'backend')
    cmd = [py, '-m', 'uvicorn', mod, '--host', '0.0.0.0', '--port', str(port)]
    p = subprocess.Popen(cmd, cwd=wd, env=env, stdout=open(os.path.join(wd, '.freebuff', f'{name}.log'), 'w'), stderr=open(os.path.join(wd, '.freebuff', f'{name}.log.err'), 'w'))
    print(f'started {name} :{port} HTTP (pid {p.pid})')
    time.sleep(1)
"@

$launchScript | & $py 2>&1 | ForEach-Object { Write-Output $_ }
Write-Output "All HTTP services starting..."
