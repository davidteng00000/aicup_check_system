import{S as se,e as ie,s as oe,F as q,G as E,w,u as I,H as N,a0 as ue,a1 as _e,Z as fe,ae as re,o as z,m as S,g as d,h as b,V as me,W as ce,r as he,v as ge,k as v,I as K,Q as de,M as L,N as A,j as O,n as D,O as p,t as be,K as J,p as j,x as ve,B as ke}from"./index-40573ec4.js";import{B as we,n as P}from"./Button-20e55939.js";import{B as Ie}from"./BlockLabel-06126da7.js";import{E as Be}from"./Empty-4e214ade.js";import{I as x}from"./Image-0cb7aa0d.js";function R(n,e,t){const l=n.slice();return l[29]=e[t][0],l[12]=e[t][1],l[31]=t,l}function T(n,e,t){const l=n.slice();return l[32]=e[t][0],l[12]=e[t][1],l[31]=t,l}function Me(n){let e,t,l,a,i,o,f=K(n[13]?n[13][1]:[]),m=[];for(let u=0;u<f.length;u+=1)m[u]=U(T(n,f,u));let c=n[4]&&n[13]&&Y(n);return{c(){e=S("div"),t=S("img"),a=z();for(let u=0;u<m.length;u+=1)m[u].c();i=z(),c&&c.c(),o=de(),d(t,"class","base-image svelte-m3v3vb"),L(t.src,l=n[13]?n[13][0].data:null)||d(t,"src",l),A(t,"fit-height",n[5]),d(e,"class","image-container svelte-m3v3vb")},m(u,h){b(u,e,h),O(e,t),O(e,a);for(let r=0;r<m.length;r+=1)m[r]&&m[r].m(e,null);b(u,i,h),c&&c.m(u,h),b(u,o,h)},p(u,h){if(h[0]&8192&&!L(t.src,l=u[13]?u[13][0].data:null)&&d(t,"src",l),h[0]&32&&A(t,"fit-height",u[5]),h[0]&24704){f=K(u[13]?u[13][1]:[]);let r;for(r=0;r<f.length;r+=1){const k=T(u,f,r);m[r]?m[r].p(k,h):(m[r]=U(k),m[r].c(),m[r].m(e,null))}for(;r<m.length;r+=1)m[r].d(1);m.length=f.length}u[4]&&u[13]?c?c.p(u,h):(c=Y(u),c.c(),c.m(o.parentNode,o)):c&&(c.d(1),c=null)},i:D,o:D,d(u){u&&(v(e),v(i),v(o)),p(m,u),c&&c.d(u)}}}function Ae(n){let e,t;return e=new Be({props:{size:"large",unpadded_box:!0,$$slots:{default:[Se]},$$scope:{ctx:n}}}),{c(){q(e.$$.fragment)},m(l,a){E(e,l,a),t=!0},p(l,a){const i={};a[1]&8&&(i.$$scope={dirty:a,ctx:l}),e.$set(i)},i(l){t||(w(e.$$.fragment,l),t=!0)},o(l){I(e.$$.fragment,l),t=!1},d(l){N(e,l)}}}function U(n){let e,t,l;return{c(){e=S("img"),d(e,"class","mask fit-height svelte-m3v3vb"),L(e.src,t=n[32].data)||d(e,"src",t),d(e,"style",l=n[7]&&n[12]in n[7]?null:`filter: hue-rotate(${Math.round(n[31]*360/n[13][1].length)}deg);`),A(e,"active",n[14]==n[12]),A(e,"inactive",n[14]!=n[12]&&n[14]!=null)},m(a,i){b(a,e,i)},p(a,i){i[0]&8192&&!L(e.src,t=a[32].data)&&d(e,"src",t),i[0]&8320&&l!==(l=a[7]&&a[12]in a[7]?null:`filter: hue-rotate(${Math.round(a[31]*360/a[13][1].length)}deg);`)&&d(e,"style",l),i[0]&24576&&A(e,"active",a[14]==a[12]),i[0]&24576&&A(e,"inactive",a[14]!=a[12]&&a[14]!=null)},d(a){a&&v(e)}}}function Y(n){let e,t=K(n[13][1]),l=[];for(let a=0;a<t.length;a+=1)l[a]=y(R(n,t,a));return{c(){e=S("div");for(let a=0;a<l.length;a+=1)l[a].c();d(e,"class","legend svelte-m3v3vb")},m(a,i){b(a,e,i);for(let o=0;o<l.length;o+=1)l[o]&&l[o].m(e,null)},p(a,i){if(i[0]&467072){t=K(a[13][1]);let o;for(o=0;o<t.length;o+=1){const f=R(a,t,o);l[o]?l[o].p(f,i):(l[o]=y(f),l[o].c(),l[o].m(e,null))}for(;o<l.length;o+=1)l[o].d(1);l.length=t.length}},d(a){a&&v(e),p(l,a)}}}function y(n){let e,t=n[12]+"",l,a,i,o;function f(){return n[24](n[12])}function m(){return n[25](n[12])}function c(){return n[28](n[31])}return{c(){e=S("div"),l=be(t),a=z(),d(e,"class","legend-item svelte-m3v3vb"),J(e,"background-color",n[7]&&n[12]in n[7]?n[7][n[12]]+"88":`hsla(${Math.round(n[31]*360/n[13][1].length)}, 100%, 50%, 0.3)`)},m(u,h){b(u,e,h),O(e,l),O(e,a),i||(o=[j(e,"mouseover",f),j(e,"focus",m),j(e,"mouseout",n[26]),j(e,"blur",n[27]),j(e,"click",c)],i=!0)},p(u,h){n=u,h[0]&8192&&t!==(t=n[12]+"")&&ve(l,t),h[0]&8320&&J(e,"background-color",n[7]&&n[12]in n[7]?n[7][n[12]]+"88":`hsla(${Math.round(n[31]*360/n[13][1].length)}, 100%, 50%, 0.3)`)},d(u){u&&v(e),i=!1,ke(o)}}}function Se(n){let e,t;return e=new x({}),{c(){q(e.$$.fragment)},m(l,a){E(e,l,a),t=!0},i(l){t||(w(e.$$.fragment,l),t=!0)},o(l){I(e.$$.fragment,l),t=!1},d(l){N(e,l)}}}function je(n){let e,t,l,a,i,o,f,m;const c=[n[11]];let u={};for(let _=0;_<c.length;_+=1)u=fe(u,c[_]);e=new re({props:u}),l=new Ie({props:{show_label:n[3],Icon:x,label:n[12]||n[15]("image.image")}});const h=[Ae,Me],r=[];function k(_,g){return _[13]==null?0:1}return o=k(n),f=r[o]=h[o](n),{c(){q(e.$$.fragment),t=z(),q(l.$$.fragment),a=z(),i=S("div"),f.c(),d(i,"class","container svelte-m3v3vb")},m(_,g){E(e,_,g),b(_,t,g),E(l,_,g),b(_,a,g),b(_,i,g),r[o].m(i,null),m=!0},p(_,g){const C=g[0]&2048?me(c,[ce(_[11])]):{};e.$set(C);const B={};g[0]&8&&(B.show_label=_[3]),g[0]&36864&&(B.label=_[12]||_[15]("image.image")),l.$set(B);let M=o;o=k(_),o===M?r[o].p(_,g):(he(),I(r[M],1,1,()=>{r[M]=null}),ge(),f=r[o],f?f.p(_,g):(f=r[o]=h[o](_),f.c()),w(f,1),f.m(i,null))},i(_){m||(w(e.$$.fragment,_),w(l.$$.fragment,_),w(f),m=!0)},o(_){I(e.$$.fragment,_),I(l.$$.fragment,_),I(f),m=!1},d(_){_&&(v(t),v(a),v(i)),N(e,_),N(l,_),r[o].d()}}}function qe(n){let e,t;return e=new we({props:{visible:n[2],elem_id:n[0],elem_classes:n[1],padding:!1,height:n[5],width:n[6],allow_overflow:!1,container:n[8],scale:n[9],min_width:n[10],$$slots:{default:[je]},$$scope:{ctx:n}}}),{c(){q(e.$$.fragment)},m(l,a){E(e,l,a),t=!0},p(l,a){const i={};a[0]&4&&(i.visible=l[2]),a[0]&1&&(i.elem_id=l[0]),a[0]&2&&(i.elem_classes=l[1]),a[0]&32&&(i.height=l[5]),a[0]&64&&(i.width=l[6]),a[0]&256&&(i.container=l[8]),a[0]&512&&(i.scale=l[9]),a[0]&1024&&(i.min_width=l[10]),a[0]&63672|a[1]&8&&(i.$$scope={dirty:a,ctx:l}),e.$set(i)},i(l){t||(w(e.$$.fragment,l),t=!0)},o(l){I(e.$$.fragment,l),t=!1},d(l){N(e,l)}}}function Ee(n,e,t){let l;ue(n,_e,s=>t(15,l=s));let{elem_id:a=""}=e,{elem_classes:i=[]}=e,{visible:o=!0}=e,{value:f}=e,m,c,{label:u=l("annotated_image.annotated_image")}=e,{show_label:h=!0}=e,{show_legend:r=!0}=e,{height:k}=e,{width:_}=e,{color_map:g}=e,{container:C=!0}=e,{scale:B=null}=e,{min_width:M=void 0}=e,{root:F}=e,{root_url:G}=e,Q=null,{loading_status:X}=e,{gradio:H}=e;function V(s){t(14,Q=s)}function W(){t(14,Q=null)}function Z(s){H.dispatch("select",{value:u,index:s})}const $=s=>V(s),ee=s=>V(s),le=()=>W(),ne=()=>W(),te=s=>Z(s);return n.$$set=s=>{"elem_id"in s&&t(0,a=s.elem_id),"elem_classes"in s&&t(1,i=s.elem_classes),"visible"in s&&t(2,o=s.visible),"value"in s&&t(19,f=s.value),"label"in s&&t(12,u=s.label),"show_label"in s&&t(3,h=s.show_label),"show_legend"in s&&t(4,r=s.show_legend),"height"in s&&t(5,k=s.height),"width"in s&&t(6,_=s.width),"color_map"in s&&t(7,g=s.color_map),"container"in s&&t(8,C=s.container),"scale"in s&&t(9,B=s.scale),"min_width"in s&&t(10,M=s.min_width),"root"in s&&t(20,F=s.root),"root_url"in s&&t(21,G=s.root_url),"loading_status"in s&&t(11,X=s.loading_status),"gradio"in s&&t(22,H=s.gradio)},n.$$.update=()=>{n.$$.dirty[0]&16252928&&(f!==m&&(t(23,m=f),H.dispatch("change")),f?t(13,c=[P(f[0],F,G),f[1].map(([s,ae])=>[P(s,F,G),ae])]):t(13,c=null))},[a,i,o,h,r,k,_,g,C,B,M,X,u,c,Q,l,V,W,Z,f,F,G,H,m,$,ee,le,ne,te]}class Ne extends se{constructor(e){super(),ie(this,e,Ee,qe,oe,{elem_id:0,elem_classes:1,visible:2,value:19,label:12,show_label:3,show_legend:4,height:5,width:6,color_map:7,container:8,scale:9,min_width:10,root:20,root_url:21,loading_status:11,gradio:22},null,[-1,-1])}}const Ke=Ne;export{Ke as default};
//# sourceMappingURL=index-bd6158ea.js.map