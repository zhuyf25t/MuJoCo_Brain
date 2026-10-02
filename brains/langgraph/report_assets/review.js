"use strict";
const data = window.REPORT;
const phases = {origin:"起源",empty:"未持球",pending:"待确认持球",holding:"已持球",final:"最终检查"};
const names = {analyze_reach:"分析 reach 参考",find_ball:"定位目标球",find_box:"定位箱口",match_grasp:"判断可抓范围",plan_motion:"规划移动",analyze_motion:"分析移动经验",check_held:"确认是否持球",match_drop:"判断投放位置",check_drop:"检查球是否入箱"};
const labels = {yes:"是",no:"否",unknown:"不确定",visible:"可见",absent:"未见",need_info:"申请资料",move:"移动"};
const $ = selector => document.querySelector(selector);
function el(tag, text, cls) {const n=document.createElement(tag);if(text!==undefined&&text!==null)n.textContent=text;if(cls)n.className=cls;return n;}
function badge(text, tone="") {return el("span",text,"badge "+tone);}
function json(value) {return el("pre",JSON.stringify(value,(key,item)=>["image_url","img_before_urls","img_after_urls"].includes(key)?undefined:item,2));}
function details(title,child,open=false) {const d=el("details");d.open=open;d.append(el("summary",title));const body=el("div",null,"content");if(child)body.append(child);d.append(body);return d;}
function notice(text,error=false) {return el("div",text,"notice"+(error?" error":""));}
function command(action) {return action.tool+"("+Object.entries(action.args||{}).map(([k,v])=>k+"="+JSON.stringify(v)).join(", ")+")";}
function inspectable(round) {return round.warnings.length||round.error||round.calls.some(c=>c.error||c.invalid_results.length||(c.result||{}).status==="unknown");}
function picture(frame,title,target,region) {
  const point=target&&(target.center||target),radius=target&&target.radius;
  const figure=el("figure"),box=el("div",null,"picture");
  if(frame&&frame.image_url){
    const link=el("a");link.href=frame.image_url;link.target="_blank";link.rel="noopener";
    const img=el("img");img.src=frame.image_url;img.alt=title;img.loading="lazy";link.append(img);box.append(link);
    if(point||region){
      const ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg");
      // Work in original pixels so a circle stays round on a non-square camera image.
      const draw=()=>{const w=img.naturalWidth,h=img.naturalHeight;if(!w||!h)return;svg.replaceChildren();svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
        if(region&&region.length){const p=document.createElementNS(ns,"polygon");p.setAttribute("points",region.map(p=>`${p.x*w},${p.y*h}`).join(" "));p.setAttribute("fill","#138bca28");p.setAttribute("stroke","#168be2");p.setAttribute("stroke-width",Math.max(1,w/300));svg.append(p);}
        if(point){const p=document.createElementNS(ns,"circle");p.setAttribute("cx",point.x*w);p.setAttribute("cy",point.y*h);p.setAttribute("r",radius?radius*w:Math.max(2,w/100));p.setAttribute("fill","none");p.setAttribute("stroke","#ef293c");p.setAttribute("stroke-width",Math.max(1,w/300));svg.append(p);}
      };img.addEventListener("load",draw);if(img.complete)draw();
      box.append(svg);
      const label=el("label",null,"toggle"),check=el("input");check.type="checkbox";check.checked=true;check.addEventListener("change",()=>svg.style.display=check.checked?"":"none");label.append(check,document.createTextNode(radius?"显示模型估计的球轮廓圆（非真值；蓝色为参考区域）":"显示定位点 / 参考区域（本记录没有球半径）"));figure.append(box,label);
    }else figure.append(box);
  }else {box.append(el("div","本轮未保存这张图片","missing"));figure.append(box);}
  figure.append(el("figcaption",title+(frame&&frame.pose?" · "+frame.pose:"")));return figure;
}
function grid(figures) {const g=el("div",null,"image-grid"+(figures.length===1?" one":""));g.append(...figures);return g;}
function moduleCard(call,index) {
  const r=call.result||{},v=call.input||{},d=el("details",null,"module"+(call.error?" error":call.warning?" warning":""));
  d.id="module-"+index;d.open=Boolean(call.error||call.warning);
  const summary=el("summary");summary.append(el("span",index+1,"module-index"),el("span",(names[call.capability]||call.capability)+" · "+call.capability,"module-title"));
  summary.append(badge(call.error?"调用失败":labels[r.status]||(r.valid===undefined?"已返回":r.valid?"分析有效":"区域 / 参照不确定"),call.error?"bad":r.status==="unknown"?"warn":""));
  if(call.seconds!==undefined)summary.append(el("span",call.seconds.toFixed(2)+" s","meta-line"));d.append(summary);
  const body=el("div",null,"content");if(call.warning)body.append(notice(call.warning));
  const profile=call.profile||{},actualModel=(call.attempts.find(a=>a.model)||{}).model||profile.model||"继承项目配置";
  body.append(el("p",`实现：${profile.backend||"llm"}  ·  模型：${actualModel}  ·  版本：${call.version||"未记录"}`,"meta-line"));
  body.append(el("h3","输入给这个模块的图片"));
  const figs=[];if(v.comparison)figs.push(picture(v.comparison,"BEFORE · 对比前图",v.comparison_target));
  const point=v.target || r.target;
  figs.push(picture(v.frame,v.comparison?"AFTER CURRENT · 本次判断图":"CURRENT · 本次判断图",point,call.capability==="analyze_reach"&&r.valid?r.region:null));body.append(grid(figs));
  const fields={};for(const k of ["task","intent","target","previous_target","comparison_target","assessment","summary","commands","max_seconds","info_feedback"]){if(v[k]!==undefined&&v[k]!==null&&v[k]!==""&&!(Array.isArray(v[k])&&!v[k].length))fields[k]=v[k];}
  body.append(details("输入的目标、任务、资料目录与动作限制",json(fields),true));
  for(const sample of v.evidence||[]){
    const b=el("div"),fs=sample.frames||[];
    b.append(grid(fs.map((f,i)=>picture(f,sample.kind==="reach"?"REFERENCE · 历史 reach 原图":`REFERENCE · ${i===0?"BEFORE 动作前":"AFTER 动作后"}`,sample.observation&&sample.observation[i===0?"before":"after"],sample.kind==="reach"&&sample.analysis.valid?sample.analysis.region:null))));
    b.append(json({sample_id:sample.id,kind:sample.kind,direction:sample.direction,commands:sample.commands,analysis:sample.analysis,image_rate:sample.image_rate,observation:sample.observation}));
    body.append(details("收到的经验样本 · "+sample.kind+" / "+(sample.direction||"reach"),b));
  }
  if(!(v.evidence||[]).length)body.append(el("p","本次调用没有附加经验样本；目录信息不等于样本内容。","muted"));
  body.append(details("初始 prompt（模块 prompt + 公共约束）",el("pre",call.prompt+"\n\n"+call.common_prompt)));
  body.append(el("h3","模块输出"));body.append(call.error?notice("调用失败："+call.error+"；没有得到可用输出。",true):json(r));
  if(call.attempts.length||call.invalid_results.length){const b=el("div");b.append(json(call.attempts));for(const invalid of call.invalid_results)b.append(el("h4","未通过校验的完整返回"),json(invalid));body.append(details("API 请求、重试、耗时及原始错误",b,Boolean(call.error)));}
  body.append(details("完整输入 JSON",json(v)));d.append(body);return d;
}
function actionsCard(round) {
  const card=el("section",null,"card");card.append(el("h3","实际执行与外部影响"),el("p","以下来自执行器记录。工具 OK 表示指令执行成功，不等于抓到了球。反馈和俯视图仅供人检查。","muted"));
  if(!round.actions.length)card.append(el("p","本轮没有执行新动作。","empty"));
  for(const action of round.actions){
    const row=el("div",null,"action"),head=el("div",null,"action-head");head.append(badge("动作 "+(action.i+1)),el("strong",command(action)),badge(action.ok?"工具 OK":"工具失败",action.ok?"":"bad"));row.append(head);
    row.append(el("p","程序执行说明："+action.thought),el("p","执行器反馈："+action.result,"tool-result"));
    const before=action.img_before_urls||{},after=action.img_after_urls||{};
    row.append(grid([picture({image_url:before.front||before.front_cam},"执行前 · 记录相机 front_cam"),picture({image_url:after.front||after.front_cam},"执行后 · 记录相机 front_cam")]));
    if(before.overhead||after.overhead)row.append(details("俯视图对照（仅调试查看）",grid([picture({image_url:before.overhead},"执行前 · overhead"),picture({image_url:after.overhead},"执行后 · overhead")])));
    row.append(details("原始动作记录",json(action)));card.append(row);
  }return card;
}
let selected=Number(location.hash.slice(1))||data.default_round||1;
function renderNav(){const nav=$("#rounds");nav.replaceChildren();for(const round of data.rounds){if($("#issues-only").checked&&!inspectable(round))continue;const btn=el("button",null,"round"+(round.number===selected?" selected":""));btn.type="button";btn.setAttribute("aria-current",round.number===selected?"step":"false");const row=el("div",null,"row");row.append(el("strong","第 "+round.number+" 轮"));if(round.error)row.append(badge("停止","bad"));else if(round.warnings.length)row.append(badge("矛盾","warn"));else if(inspectable(round))row.append(badge("需复核","warn"));btn.append(row,el("small",phases[round.phase_before]+" → "+phases[round.phase_after]),el("small",round.actions.length?round.actions.map(a=>command(a)).join(" → "):"无新动作"));btn.addEventListener("click",()=>select(round.number));nav.append(btn);}}
function select(number){selected=number;history.replaceState(null,"","#"+number);renderNav();renderRound();}
function renderRound(){
  const round=data.rounds.find(r=>r.number===selected)||data.rounds[0],main=$("#detail");main.replaceChildren();if(!round){main.append(el("p","没有找到决策记录。"));return;}
  const heading=el("div",null,"round-title");heading.append(el("h2",`第 ${round.number} 轮 · ${phases[round.phase_before]} → ${phases[round.phase_after]}`));const controls=el("div",null,"step-controls");
  for(const [delta,label] of [[-1,"← 上一轮"],[1,"下一轮 →"]]){const b=el("button",label);b.type="button";b.disabled=!data.rounds.some(r=>r.number===round.number+delta);b.addEventListener("click",()=>select(round.number+delta));controls.append(b);}heading.append(controls);main.append(heading);
  main.append(el("p",`${round.calls.length} 次模块调用 · ${round.actions.length} 条实际指令 · 本轮图像 ${(round.frame||{}).id||"未记录"}`,"meta-line"));
  for(const warning of round.warnings)main.append(notice(warning));if(round.error)main.append(notice("本轮停止原因："+round.error.error+": "+round.error.message,true));
  const hero=el("section",null,"card hero"),photo=el("div"),summary=el("div");
  photo.append(el("h3","这一轮开始时，车头看到了什么"));const detection=[...round.calls].reverse().find(c=>["find_ball","find_box"].includes(c.capability)&&c.result&&c.result.target);photo.append(picture(round.frame,"保存的原始车头图；点击图片可查看原图",detection&&detection.result.target));
  summary.append(el("h3","这一轮发生了什么"));const flow=el("div",null,"flow");round.calls.forEach((c,i)=>{if(i)flow.append(el("span","→","arrow"));const a=el("a",names[c.capability]||c.capability);a.href="#module-"+i;a.addEventListener("click",e=>{e.preventDefault();const target=$("#module-"+i);target.open=true;target.scrollIntoView({behavior:"smooth",block:"start"});});flow.append(a);});summary.append(flow);
  for(const call of round.calls){const r=call.result||{};summary.append(el("p",`${names[call.capability]||call.capability}：${call.error?"调用失败":labels[r.status]||(r.valid===undefined?"已返回":r.valid?"分析有效":"不确定")}`));}
  const probes=round.info.filter(e=>e.event==="probe");if(probes.length)summary.append(notice("经验库缺少所需方向的数据，程序只执行一次小步试探。模型提议的时长可能没有直接执行。"));
  summary.append(el("h4","最终实际动作"),el("p",round.actions.length?round.actions.map(command).join(" → "):"无"));hero.append(photo,summary);main.append(hero);
  if(round.info.length){const section=el("section",null,"card");section.append(el("h3","数据库查询与程序处理"));for(const event of round.info){let line=event.event;if(event.event==="request_info")line=`申请 ${event.request.kind}/${event.request.direction||"reach"}：${event.hit?"命中":"缺数据"}，来源 ${event.source}`;else if(event.event==="probe")line="安排一次试探："+event.request.direction;else if(event.event==="motion_sample")line="本轮前后图经验："+(event.accepted?"已收录":"未收录（参照不可靠）");else if(event.event==="cache_hit")line="复用同一输入的模块结果："+event.capability;else if(event.event==="reference_reanalyzed")line="用保存的 reach 原图重新分析";else if(event.event==="repeated_request")line="模型重复申请已有资料，程序返回资料已齐的反馈";section.append(el("p",line));}section.append(details("查询及处理的完整记录",json(round.info)));main.append(section);}
  const modules=el("section",null,"card");modules.append(el("h3","按顺序展开模块调用"),el("p","每个模块分别展示实际收到的图片、资料、prompt，以及返回结果。","muted"));round.calls.forEach((c,i)=>modules.append(moduleCard(c,i)));if(!round.calls.length)modules.append(el("p","本轮由程序执行固定动作，没有调用模型。","empty"));main.append(modules,actionsCard(round),details("本轮完整事件记录",json(round.events)));
}
$("#task").textContent=data.meta.task||"未记录任务";const stopped=!data.meta.success;$("#run-state").textContent=data.in_progress?"运行中 · 当前快照":stopped?"已结束 · 未完成任务":"已结束 · 任务成功";$("#run-state").className="badge"+(stopped&&!data.in_progress?" bad":"");
const wall=data.meta.t_wall_end&&data.meta.t_wall_start?Math.round(data.meta.t_wall_end-data.meta.t_wall_start):null;
for(const [value,label] of [[data.stats.rounds,"决策轮"],[data.stats.actions,"实际动作"],[data.stats.api_calls,"API 请求"],[data.stats.api_seconds+"s","API 等待"],[wall===null?"—":wall+"s","运行耗时"],[data.meta.t_sim===undefined?"—":data.meta.t_sim+"s","仿真时间"]]){const m=el("div",null,"metric");m.append(el("b",value),document.createTextNode(label));$("#overview").append(m);}
const ending=el("div",null,"metric");ending.style.flexBasis="100%";ending.textContent=data.in_progress?"运行尚未结束，这是生成时的快照；结束后会自动重建，刷新查看完整记录。":"最终停止原因："+(data.meta.reason||"未记录")+"。可从第 "+data.default_round+" 轮开始复核；标记是模型输出，不保证正确。";$("#overview").append(ending);
$("#issues-only").addEventListener("change",renderNav);window.addEventListener("hashchange",()=>{const n=Number(location.hash.slice(1));if(data.rounds.some(r=>r.number===n)){selected=n;renderNav();renderRound();}});renderNav();renderRound();
