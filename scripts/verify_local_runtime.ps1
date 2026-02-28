Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$r = Invoke-RestMethod -Method GET "http://127.0.0.1:8000/v1/system/runtime"
$s = Invoke-RestMethod -Method GET "http://127.0.0.1:8000/v1/system/status"

function Get-OptionalProperty {
  param(
    [object]$ObjectValue,
    [string]$PropertyName,
    [string]$DefaultValue = "n/a"
  )
  if ($null -eq $ObjectValue) {
    return $DefaultValue
  }
  $prop = $ObjectValue.PSObject.Properties[$PropertyName]
  if ($null -eq $prop) {
    return $DefaultValue
  }
  if ($null -eq $prop.Value) {
    return $DefaultValue
  }
  return [string]$prop.Value
}

@"
backend_configured=$($r.model_backend_configured)
backend_effective=$($r.model_backend_effective)
judge_enabled=$($r.response_match_model_enabled)
model_ready=$($r.model_ready)
memory_ready=$($s.memory_ready)
privacy_redaction_enabled=$(Get-OptionalProperty -ObjectValue $r -PropertyName "privacy_redaction_enabled")
privacy_remove_sensitive_context_keys=$(Get-OptionalProperty -ObjectValue $r -PropertyName "privacy_remove_sensitive_context_keys")
relationship_memory_text_enabled=$(Get-OptionalProperty -ObjectValue $r -PropertyName "relationship_memory_text_enabled")
fallback_rate=$($s.telemetry.fallback_rate)
"@
