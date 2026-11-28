"""Lua scripts shared by every consumer of a capped Redis stream (workers, API persister).

See ``worker/streams.py`` for why the read must be one atomic script: it is the only way
to count, exactly, entries that ``MAXLEN`` trimmed before any consumer read them.
"""

# KEYS: stream, metrics hash. ARGV: group, consumer, count.
# Returns {entries or false, skipped, last-delivered-id}.
READ_SCRIPT = r"""
local function group_info()
  for _, g in ipairs(redis.call('XINFO', 'GROUPS', KEYS[1])) do
    local info = {}
    for i = 1, #g, 2 do info[g[i]] = g[i + 1] end
    if info['name'] == ARGV[1] then return info end
  end
  return redis.error_reply('NOGROUP no group ' .. ARGV[1])
end
local before = group_info()['entries-read'] or 0
local res = redis.call('XREADGROUP', 'GROUP', ARGV[1], ARGV[2], 'COUNT', ARGV[3],
                       'STREAMS', KEYS[1], '>')
local n = 0
if res then n = #res[1][2] end
local after = group_info()
local skipped = (after['entries-read'] or 0) - before - n
if skipped > 0 then
  redis.call('HINCRBY', KEYS[2], 'lost_trimmed_undelivered', skipped)
end
if n > 0 then redis.call('HINCRBY', KEYS[2], 'delivered', n) end
return {res, skipped, after['last-delivered-id']}
"""
