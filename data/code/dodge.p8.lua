-- dodge -- a tiny PICO-8-style arcade game (original, MIT-licensed sample)
-- Plain Lua extracted from the cart's __lua__ section. Used as a small, self
-- contained code corpus for code-navigation / explanation / decomposition tasks.
-- ~10 functions with clear call sites so tasks have deterministic gold answers.

player = {}
enemies = {}
score = 0
lives = 3
state = "play"

function _init()
  reset_game()
end

function reset_game()
  player = { x = 64, y = 110, w = 6, spd = 2 }
  enemies = {}
  score = 0
  lives = 3
  state = "play"
end

function spawn_enemy()
  local e = { x = flr(rnd(120)) + 4, y = -4, spd = rnd(2) + 1 }
  add(enemies, e)
end

function move_player()
  if btn(0) then player.x = player.x - player.spd end
  if btn(1) then player.x = player.x + player.spd end
  player.x = mid(2, player.x, 120)
end

function update_enemies()
  for e in all(enemies) do
    e.y = e.y + e.spd
    if e.y > 132 then
      del(enemies, e)
      score = score + 1
    end
  end
end

function check_collision()
  for e in all(enemies) do
    if abs(e.x - player.x) < 6 and abs(e.y - player.y) < 6 then
      del(enemies, e)
      lose_life()
    end
  end
end

function lose_life()
  lives = lives - 1
  if lives <= 0 then
    state = "over"
  end
end

function draw_hud()
  print("score " .. score, 2, 2, 7)
  print("lives " .. lives, 90, 2, 8)
end

function _update()
  if state == "over" then
    if btnp(4) then reset_game() end
    return
  end
  move_player()
  update_enemies()
  check_collision()
  if rnd(1) < 0.04 then spawn_enemy() end
end

function _draw()
  cls(1)
  spr(1, player.x, player.y)
  for e in all(enemies) do
    spr(2, e.x, e.y)
  end
  draw_hud()
  if state == "over" then
    print("game over", 44, 60, 7)
  end
end
