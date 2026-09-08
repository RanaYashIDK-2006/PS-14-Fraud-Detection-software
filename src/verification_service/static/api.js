// api.js — API client generated from a FastAPI OpenAPI schema at runtime.
//
// Every endpoint the server advertises becomes a callable
// `client.<operationId>` (camelCased) with the correct method, path/query
// parameters, and request body — so a NEW backend endpoint is automatically
// callable from the UI with no hand-written fetch wrapper. This file is the
// single place that knows how to speak HTTP to the services; index.html only
// knows operation names.
//
//   createClient(spec, {
//     base,                  // service base URL
//     authMode(opId),        // "bearer" | "compliance" | "none"  (default "bearer")
//     getToken(),            // current Bearer token, read PER REQUEST
//     getComplianceToken(),  // compliance passphrase, read PER REQUEST
//   })
//
//   client.confirmAlert({ event_id: "...", body: { outcome: "this_was_me" } })
//   client.complianceEvents({ limit: 40, offset: 0 })
//   client.overview()
//
// args: path + query parameters by name, plus `body` for the payload.
// Errors: throws `Error(detail || "HTTP <status>")` — the same contract the
// old hand-written `api()` helper used, so `.catch((e) => ...)` call sites
// are unchanged.
(function (root) {
  "use strict";

  const METHODS = /^(get|post|put|patch|delete)$/;

  function camel(opId) {
    return opId.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
  }

  function createClient(spec, opts) {
    const base = opts.base;
    const authMode = opts.authMode || (() => "bearer");
    const getToken = opts.getToken || (() => "");
    const getComplianceToken = opts.getComplianceToken || (() => "");
    const client = {};

    for (const [path, methods] of Object.entries(spec.paths || {})) {
      for (const [method, op] of Object.entries(methods)) {
        if (!METHODS.test(method) || !op.operationId) continue;
        const name = camel(op.operationId);
        if (client[name]) continue; // duplicate operationId: first wins
        const queryParams = (op.parameters || [])
          .filter((p) => p.in === "query")
          .map((p) => p.name);

        client[name] = async (args) => {
          args = args || {};
          let url = base + path.replace(/\{([^}]+)\}/g, (_, name) =>
            encodeURIComponent(String(args[name] == null ? "" : args[name])));
          const qs = queryParams
            .filter((n) => args[n] !== undefined)
            .map((n) => encodeURIComponent(n) + "=" + encodeURIComponent(args[n]))
            .join("&");
          if (qs) url += (url.includes("?") ? "&" : "?") + qs;

          const headers = { "Content-Type": "application/json" };
          const mode = authMode(name);
          if (mode === "bearer" && getToken()) headers["Authorization"] = "Bearer " + getToken();
          if (mode === "compliance" && getComplianceToken()) headers["X-Compliance-Token"] = getComplianceToken();

          const init = { method: method.toUpperCase(), headers };
          if (op.requestBody) init.body = JSON.stringify(args.body === undefined ? {} : args.body);

          const res = await fetch(url, init);
          const data = await res.json().catch(() => ({}));
          if (!res.ok) {
            const err = new Error(data.detail || ("HTTP " + res.status));
            err.status = res.status;  // lets the UI detect 401 without string-matching
            throw err;
          }
          return data;
        };
      }
    }
    return client;
  }

  root.createClient = createClient;
  if (typeof module !== "undefined" && module.exports) module.exports = { createClient };
})(typeof window !== "undefined" ? window : globalThis);
